"""
Visualize VLA vision features on input images.

This utility provides a lightweight way to inspect how VLM visual tokens
activate over the original image for two policies:
1) OpenVLA / OpenVLA-OFT
2) PI0.5 (LeRobot PI05)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_uint8_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    return arr


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_min = float(x.min())
    x_max = float(x.max())
    return (x - x_min) / max(x_max - x_min, eps)


def _infer_square_token_map(
    token_features: torch.Tensor, expected_num_patches: int | None = None
) -> torch.Tensor:
    """
    Convert token features [N, D] or [B, N, D] to [H, W] score map.
    """
    if token_features.dim() == 3:
        token_features = token_features[0]
    if token_features.dim() != 2:
        raise ValueError(f"Expected token features [N, D], got {tuple(token_features.shape)}")

    tokens = token_features
    n_tokens = tokens.shape[0]

    if expected_num_patches is not None and n_tokens >= expected_num_patches:
        if n_tokens != expected_num_patches:
            tokens = tokens[-expected_num_patches:]
        n_tokens = tokens.shape[0]

    n = int(np.sqrt(n_tokens))
    if n * n != n_tokens:
        n_minus_1 = n_tokens - 1
        m = int(np.sqrt(max(n_minus_1, 0)))
        if m * m == n_minus_1:
            tokens = tokens[1:]
            n_tokens = tokens.shape[0]
            n = int(np.sqrt(n_tokens))
        else:
            raise ValueError(
                f"Cannot reshape token count {n_tokens} into square grid. "
                f"Try passing expected_num_patches."
            )

    token_scores = torch.linalg.vector_norm(tokens, ord=2, dim=-1)
    return token_scores.view(n, n)


def _resize_heatmap_to_image(heatmap_hw: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    ten = torch.from_numpy(heatmap_hw)[None, None].float()
    ten = F.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)
    return ten[0, 0].cpu().numpy()


def _colorize_heatmap(heatmap_01: np.ndarray) -> np.ndarray:
    """
    Fast blue->red color ramp without matplotlib dependency.
    """
    heat = np.clip(heatmap_01, 0.0, 1.0)
    r = (255.0 * heat).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(heat - 0.5) * 2.0) * 0.75).astype(np.uint8)
    b = (255.0 * (1.0 - heat)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def make_overlay(image_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = _colorize_heatmap(heatmap_01)
    base = image_rgb.astype(np.float32)
    h = colored.astype(np.float32)
    out = np.clip((1.0 - alpha) * base + alpha * h, 0, 255).astype(np.uint8)
    return out


def save_feature_visualization(
    image_rgb: np.ndarray,
    token_features: torch.Tensor,
    output_dir: str | Path,
    prefix: str,
    expected_num_patches: int | None = None,
    alpha: float = 0.45,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_map = _infer_square_token_map(token_features, expected_num_patches=expected_num_patches)
    token_map_np = token_map.detach().float().cpu().numpy()

    full_map = _resize_heatmap_to_image(token_map_np, image_rgb.shape[:2])
    full_map_01 = _normalize_01(full_map)

    heat_u8 = (full_map_01 * 255.0).astype(np.uint8)
    overlay = make_overlay(image_rgb=image_rgb, heatmap_01=full_map_01, alpha=alpha)

    heat_path = output_dir / f"{prefix}_heatmap.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    npy_path = output_dir / f"{prefix}_token_map.npy"

    Image.fromarray(heat_u8, mode="L").save(heat_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    np.save(npy_path, token_map_np)

    return {
        "heatmap": str(heat_path),
        "overlay": str(overlay_path),
        "token_map_npy": str(npy_path),
    }


@dataclass
class OpenVLAFeatureExtractor:
    model: Any
    device: torch.device

    @staticmethod
    def from_checkpoint(
        checkpoint: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> "OpenVLAFeatureExtractor":
        from transformers import AutoModelForVision2Seq

        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        if not load_in_8bit and not load_in_4bit:
            model = model.to(torch_device)
        return OpenVLAFeatureExtractor(model=model, device=torch_device)

    def _vision_backbone(self):
        if hasattr(self.model, "vision_backbone"):
            return self.model.vision_backbone
        if hasattr(self.model, "model") and hasattr(self.model.model, "vision_backbone"):
            return self.model.model.vision_backbone
        raise AttributeError("Cannot find `vision_backbone` on loaded OpenVLA model.")

    @torch.no_grad()
    def extract_token_features(self, image: Image.Image) -> tuple[torch.Tensor, int | None]:
        vision = self._vision_backbone()
        image_transform = vision.image_transform
        pixel_values = image_transform(image.convert("RGB"))

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None].to(self.device)
            token_features = vision(pixel_values)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None].to(self.device) for k, v in pixel_values.items()}
            token_features = vision(pixel_values)
        else:
            raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)}")

        expected = getattr(vision, "num_patches", None)
        return token_features, expected


@dataclass
class PI05FeatureExtractor:
    policy: Any
    device: torch.device

    @staticmethod
    def _ensure_lerobot_path(repo_root: Path) -> None:
        candidates = [
            repo_root / "core" / "policies" / "pi05" / "Lerobot" / "src",
            repo_root.parent / "lerobot" / "src",
        ]
        for c in candidates:
            if c.is_dir():
                c_str = str(c)
                if c_str not in sys.path:
                    sys.path.insert(0, c_str)

    @staticmethod
    def from_checkpoint(checkpoint: str, device: str = "cuda") -> "PI05FeatureExtractor":
        repo_root = Path(__file__).resolve().parents[1]
        PI05FeatureExtractor._ensure_lerobot_path(repo_root)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = str(torch_device)
        policy = PI05Policy.from_pretrained(checkpoint, config=cfg).to(torch_device).eval()
        return PI05FeatureExtractor(policy=policy, device=torch_device)

    @torch.no_grad()
    def extract_token_features(self, image: Image.Image) -> tuple[torch.Tensor, int | None]:
        # PI05 preprocessing: [0,1] -> resize/pad -> [-1,1]
        model = self.policy.model
        target_h, target_w = self.policy.config.image_resolution
        img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        ten = torch.from_numpy(img).to(self.device)[None]  # [1,H,W,C]
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if ten.shape[1:3] != (target_h, target_w):
            ten = resize_with_pad_torch(ten, target_h, target_w)
        ten = ten * 2.0 - 1.0
        ten = ten.permute(0, 3, 1, 2).contiguous()  # [1,3,H,W]

        token_features = model.paligemma_with_expert.embed_image(ten)
        return token_features, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VLA feature visualization")
    p.add_argument("--policy", type=str, choices=["openvla", "pi05"], required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    p.add_argument("--image", type=str, required=True, help="Input RGB image path")
    p.add_argument("--output-dir", type=str, default="feature_viz_outputs")
    p.add_argument("--prefix", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    image_rgb = _to_uint8_rgb(image)
    prefix = args.prefix or args.policy

    if args.policy == "openvla":
        extractor = OpenVLAFeatureExtractor.from_checkpoint(
            checkpoint=args.checkpoint,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        extractor = PI05FeatureExtractor.from_checkpoint(
            checkpoint=args.checkpoint,
            device=args.device,
        )

    token_features, expected_num_patches = extractor.extract_token_features(image)
    paths = save_feature_visualization(
        image_rgb=image_rgb,
        token_features=token_features,
        output_dir=args.output_dir,
        prefix=prefix,
        expected_num_patches=expected_num_patches,
        alpha=args.alpha,
    )

    print("Saved feature visualization:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
