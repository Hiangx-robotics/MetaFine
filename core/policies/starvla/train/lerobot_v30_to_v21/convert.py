#!/usr/bin/env python3
"""LeRobot Dataset v3.0 -> v2.1 Converter

Wrapper script that converts one or more LeRobot v3.0 datasets to v2.1 format.
Output directories are organized as: <output-dir>/<robot_type>/<dataset_id>/

Usage:
    # Convert a single dataset
    python convert.py --input /path/to/lerobot_v30/<dataset_id> --output-dir /path/to/output

    # Batch convert all datasets under a directory
    python convert.py --input /path/to/lerobot_v30 --output-dir /path/to/output --batch

    # Disable robot_type grouping (flat output)
    python convert.py --input /path/to/lerobot_v30 --output-dir /path/to/output --batch --no-group-by-robot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from convert_dataset_v30_to_v21 import convert_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataset_info(path: Path) -> dict | None:
    """Load info.json from a dataset directory. Returns None if not found or invalid."""
    info_json = path / "meta" / "info.json"
    if not info_json.exists():
        return None
    try:
        with open(info_json) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def is_v30_dataset(path: Path) -> bool:
    """Check if a directory looks like a valid v3.0 dataset."""
    info = load_dataset_info(path)
    return info is not None and info.get("codebase_version") == "v3.0"


def get_robot_type(path: Path) -> str:
    """Extract robot_type from a dataset's info.json. Returns 'unknown' if not found."""
    info = load_dataset_info(path)
    if info is None:
        return "unknown"
    return info.get("robot_type", "unknown")


def discover_datasets(input_dir: Path) -> list[Path]:
    """Discover all v3.0 datasets under a directory."""
    datasets = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and is_v30_dataset(child):
            datasets.append(child)
    return datasets


def convert_single(input_path: Path, output_dir: Path, repo_id_prefix: str, group_by_robot: bool = True) -> bool:
    """Convert a single dataset. Returns True on success."""
    dataset_id = input_path.name
    robot_type = get_robot_type(input_path) if group_by_robot else None

    if robot_type:
        output_path = output_dir / dataset_id
    else:
        output_path = output_dir / dataset_id

    repo_id = f"{repo_id_prefix}/{dataset_id}"

    logger.info("=" * 60)
    logger.info("Dataset ID : %s", dataset_id)
    if robot_type:
        logger.info("Robot Type : %s", robot_type)
    logger.info("Input      : %s", input_path)
    logger.info("Output     : %s", output_path)
    logger.info("=" * 60)

    start_time = time.time()
    try:
        convert_dataset(
            repo_id=repo_id,
            root=str(input_path),
            output_root=str(output_path),
        )
        elapsed = time.time() - start_time
        logger.info("Done [%s] in %.1f seconds", dataset_id, elapsed)
        return True
    except Exception:
        elapsed = time.time() - start_time
        logger.exception("Failed [%s] after %.1f seconds", dataset_id, elapsed)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset(s) from v3.0 to v2.1 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Single dataset (output: <output-dir>/astribot_s1/8d85f98d.../...)
  python convert.py \\
    --input /data/lerobot_v30/8d85f98d687942d28af78efea1257f32 \\
    --output-dir /data/lerobot_v21

  # Batch convert all datasets
  python convert.py \\
    --input /data/lerobot_v30 \\
    --output-dir /data/lerobot_v21 \\
    --batch

  # Flat output without robot_type grouping
  python convert.py \\
    --input /data/lerobot_v30 \\
    --output-dir /data/lerobot_v21 \\
    --batch --no-group-by-robot
""",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single v3.0 dataset directory, or a parent directory containing multiple datasets (with --batch).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output base directory. Each dataset will be saved as <output-dir>/<dataset_id>/.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: scan --input for all v3.0 datasets and convert each one.",
    )
    parser.add_argument(
        "--repo-id-prefix",
        type=str,
        default="astribot",
        help="Repo ID prefix for the dataset (default: astribot).",
    )
    parser.add_argument(
        "--no-group-by-robot",
        action="store_true",
        help="Disable grouping by robot_type. Output directly as <output-dir>/<dataset_id>/.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    group_by_robot = not args.no_group_by_robot
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        # Batch mode: scan for datasets
        datasets = discover_datasets(input_path)
        if not datasets:
            logger.error("No v3.0 datasets found under: %s", input_path)
            sys.exit(1)

        logger.info("Found %d v3.0 datasets to convert", len(datasets))
        succeeded, failed = 0, 0

        for i, ds_path in enumerate(datasets, 1):
            logger.info("[%d/%d] Converting %s ...", i, len(datasets), ds_path.name)
            if convert_single(ds_path, output_dir, args.repo_id_prefix, group_by_robot):
                succeeded += 1
            else:
                failed += 1

        logger.info("=" * 60)
        logger.info("Batch complete: %d succeeded, %d failed, %d total", succeeded, failed, len(datasets))
        if failed > 0:
            sys.exit(1)
    else:
        # Single mode
        if not is_v30_dataset(input_path):
            logger.error("Not a valid v3.0 dataset: %s", input_path)
            sys.exit(1)

        if not convert_single(input_path, output_dir, args.repo_id_prefix, group_by_robot):
            sys.exit(1)

    logger.info("All done.")


if __name__ == "__main__":
    main()
