"""Convert MetaFine ManiSkill replay'd HDF5 trajectories to RLDS (TFDS) format.

Input: ManiSkill replay'd HDF5 files (trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5)

Usage:
  python utils/convert_to_rlds.py \
    -i /nat/demos/grasp_part/grasp_part_1 \
    -o /nat/demos/datasets/rlds \
    --dataset-name grasp_part_rlds \
    --image-size 256

  # Then train:
  DATASET_NAME=grasp_part_rlds bash core/policies/openvla-oft/train_rlds.sh
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before any TF / TFDS import to prevent GCS network requests
os.environ.setdefault("TFDS_OFFLINE", "1")
os.environ.setdefault("NO_GCE_CHECK", "true")
os.environ.setdefault("GCS_READ_CACHE_DISABLED", "1")

REPLAY_PATTERN = "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"


def main():
    parser = argparse.ArgumentParser(description="Convert MetaFine HDF5 to RLDS (TFDS)")
    parser.add_argument("-i", "--input-dirs", nargs="+", required=True,
                        help="Directories containing replay'd HDF5 files")
    parser.add_argument("-o", "--output-dir", type=str, default="/nat/demos/datasets/rlds",
                        help="Root output dir (= data_root_dir for finetune.py)")
    parser.add_argument("--dataset-name", type=str, default="fgmanip_rlds")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Override language instruction written into RLDS for every episode.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-episodes", action="store_true",
                        help="Include failed episodes (default: only successful)")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to convert across all HDF5 files. "
                             "0 means no limit (original behavior).")
    parser.add_argument("-p", "--pattern", type=str, default=REPLAY_PATTERN,
                        help="HDF5 filename pattern to search for")
    args = parser.parse_args()
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be >= 0")

    # Disable GPU for TF (we only need CPU for data conversion)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    h5_files = []
    for d in args.input_dirs:
        h5_files.extend(sorted(Path(d).rglob(args.pattern)))
    print(f"Found {len(h5_files)} replay'd HDF5 files")
    if not h5_files:
        print("Nothing to do. Check --input-dirs and --pattern.")
        return

    # Import builder and configure it
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
    from fgmanip_rlds import fgmanip_rlds_dataset_builder as builder_mod

    builder_mod.H5_FILES = h5_files
    builder_mod.IMAGE_SIZE = args.image_size
    builder_mod.VAL_RATIO = args.val_ratio
    builder_mod.SEED = args.seed
    builder_mod.ONLY_SUCCESS = not args.all_episodes
    builder_mod.DATASET_NAME = args.dataset_name
    builder_mod.INSTRUCTION_OVERRIDE = args.instruction.strip() if args.instruction else None
    if builder_mod.INSTRUCTION_OVERRIDE:
        print(f"Using instruction override: {builder_mod.INSTRUCTION_OVERRIDE}")
    if args.max_episodes != 0:
        print(f"Limiting conversion to first {args.max_episodes} episodes in total.")
        original_extract_episodes = builder_mod.extract_episodes
        converted_counter = {"count": 0}

        def _extract_episodes_with_limit(h5_path, image_size, only_success=True):
            if converted_counter["count"] >= args.max_episodes:
                return
            for ep in original_extract_episodes(h5_path, image_size, only_success):
                if converted_counter["count"] >= args.max_episodes:
                    break
                converted_counter["count"] += 1
                yield ep

        builder_mod.extract_episodes = _extract_episodes_with_limit

    builder = builder_mod.FgmanipRlds(data_dir=args.output_dir)
    dl_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder.download_and_prepare(download_config=dl_config)

    info = builder.info
    print(f"\nDataset built at: {builder.data_dir}")
    for name, split in info.splits.items():
        print(f"  {name}: {split.num_examples} episodes")
    print(f"\nTrain with:")
    print(f"  --data_root_dir {args.output_dir} --dataset_name {args.dataset_name}")


if __name__ == "__main__":
    main()
