"""TFDS dataset builder for FGManip -> RLDS conversion."""

import json
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARM_JOINTS = slice(0, 7)
GRIPPER_JOINTS = slice(7, 9)

TASK_DESCRIPTIONS = {
    "align_to_part": "align the gripper to the handle",
    "draw_triangle": "draw a triangle connecting the vertices",
    "grasp_part": "grasp the handle",
    "lid_opening": "open the lid of the bottle",
    "peg_in_hole": "pick up the peg and insert it into the box with a hole",
    "plug_charger": "pick up the charger and plug it into the receptacle",
    "press_switch": "press the switch",
    "rotate": "rotate the object part",
    "slide_along": "slide the object",
    "stand_up": "pick up the object and make it stand up",
    "toggle_switch": "toggle the switch",
    "toggle_switch_table": "toggle the switch",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_instruction(h5_path: Path) -> str:
    json_path = h5_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
            env_id = meta.get("env_info", {}).get("env_id")
            if isinstance(env_id, str):
                return TASK_DESCRIPTIONS.get(env_id, env_id.replace("_", " "))
        except Exception:
            pass
    for parent in [h5_path.parent, h5_path.parent.parent]:
        for key in TASK_DESCRIPTIONS:
            if key in parent.name:
                return TASK_DESCRIPTIONS[key]
    return h5_path.parent.name.replace("_", " ")


def _make_proprio(qpos: np.ndarray) -> np.ndarray:
    """(T, 9) qpos -> (T, 8) proprio: 7 arm joints + 1 mean gripper."""
    arm = qpos[:, ARM_JOINTS]
    gripper = qpos[:, GRIPPER_JOINTS].mean(axis=1, keepdims=True)
    return np.concatenate([arm, gripper], axis=1).astype(np.float32)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size), Image.BICUBIC))


def extract_episodes(h5_path: Path, image_size: int, only_success: bool = True):
    """Yield episode dicts from one replay'd HDF5 file."""
    instruction = INSTRUCTION_OVERRIDE.strip() if INSTRUCTION_OVERRIDE else _infer_instruction(h5_path)

    json_path = h5_path.with_suffix(".json")
    success_set = None
    if only_success and json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
            success_set = {
                ep["episode_id"]
                for ep in meta.get("episodes", [])
                if ep.get("success", False)
            }
        except Exception:
            pass

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[1]),
        )
        if not traj_keys:
            return

        for traj_key in tqdm(traj_keys, desc=f"  Episodes ({h5_path.name})", leave=False):
            ep_id = int(traj_key.split("_")[1])
            if success_set is not None and ep_id not in success_set:
                continue

            traj = f[traj_key]
            if "actions" not in traj:
                continue

            actions = np.asarray(traj["actions"][()], dtype=np.float32)
            num_steps = len(actions)
            if num_steps == 0:
                continue

            qpos = np.asarray(traj["obs"]["agent"]["qpos"][:num_steps], dtype=np.float32)
            proprio = _make_proprio(qpos)

            rgb = traj["obs"]["sensor_data"]["base_camera"]["rgb"]
            images = [_resize(rgb[t], image_size) for t in range(num_steps)]

            yield {
                "instruction": instruction,
                "actions": actions,
                "proprio": proprio,
                "images": images,
                "num_steps": num_steps,
                "source": str(h5_path),
            }


# ---------------------------------------------------------------------------
# Module-level config (set by convert_to_rlds.py before building)
# ---------------------------------------------------------------------------

H5_FILES: List[Path] = []
IMAGE_SIZE: int = 256
VAL_RATIO: float = 0.05
SEED: int = 42
ONLY_SUCCESS: bool = True
DATASET_NAME: str = "fgmanip_rlds"
INSTRUCTION_OVERRIDE: str | None = None


# ---------------------------------------------------------------------------
# TFDS Builder
# ---------------------------------------------------------------------------

class FgmanipRlds(tfds.core.GeneratorBasedBuilder):
    """FGManip RLDS dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    @property
    def name(self) -> str:
        return DATASET_NAME

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Third-person camera (base_camera) RGB.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,), dtype=np.float32,
                            doc="7 arm joint angles + 1 mean gripper width.",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(8,), dtype=np.float32,
                        doc="7 joint delta pos + 1 gripper.",
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        all_episodes = []
        for h5_path in tqdm(H5_FILES, desc="Loading HDF5 files"):
            eps_in_file = list(extract_episodes(h5_path, IMAGE_SIZE, ONLY_SUCCESS))
            all_episodes.extend(eps_in_file)
            print(f"  {h5_path.name}: {len(eps_in_file)} episodes extracted, "
                  f"total so far: {len(all_episodes)}", flush=True)

        if not all_episodes:
            raise ValueError("No episodes found. Check --input-dirs and file pattern.")

        rng = np.random.default_rng(SEED)
        indices = np.arange(len(all_episodes))
        rng.shuffle(indices)

        n_val = max(1, int(len(all_episodes) * VAL_RATIO)) if VAL_RATIO > 0 else 0
        val_set = set(indices[:n_val].tolist()) if n_val > 0 else set()

        train_eps = [ep for i, ep in enumerate(all_episodes) if i not in val_set]
        val_eps = [ep for i, ep in enumerate(all_episodes) if i in val_set]

        print(f"Total: {len(all_episodes)} episodes | train: {len(train_eps)}, val: {len(val_eps)}")

        splits = {"train": self._generate_examples(train_eps)}
        if val_eps:
            splits["val"] = self._generate_examples(val_eps)
        return splits

    def _generate_examples(self, episodes) -> Iterator[Tuple[str, Any]]:
        for ep_idx, ep in enumerate(episodes):
            steps = []
            for t in range(ep["num_steps"]):
                steps.append({
                    "observation": {
                        "image": ep["images"][t],
                        "state": ep["proprio"][t],
                    },
                    "action": ep["actions"][t],
                    "discount": 1.0,
                    "reward": float(t == ep["num_steps"] - 1),
                    "is_first": t == 0,
                    "is_last": t == ep["num_steps"] - 1,
                    "is_terminal": t == ep["num_steps"] - 1,
                    "language_instruction": ep["instruction"],
                })
            yield str(ep_idx), {
                "steps": steps,
                "episode_metadata": {"file_path": ep.get("source", "")},
            }
