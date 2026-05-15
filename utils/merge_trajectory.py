import argparse
from pathlib import Path
import h5py
from mani_skill.utils.logging_utils import logger
import os.path as osp
from mani_skill.utils.io_utils import dump_json, load_json


# python -m mani_skill.trajectory.merge_trajectory \
#    -i  /home/robot/workspace/FGManip/demos \
#    -o output.h5 \
#    -p trajectory.h5

def merge_trajectories(output_path: str, traj_paths: list, recompute_id: bool = True, success: bool = True):
    """
    Merges multiple JSON and H5 files into a single JSON and H5 file.

    This function combines the contents of multiple JSON and H5 files. It keeps the first value for all keys
    (other than "episodes") and logs a warning for any differences. The "episodes" from each JSON file are merged
    into a single list, and the corresponding H5 data is copied to the output H5 file.

    Args:
        output_path (str): The path to the output H5 file. The corresponding JSON file will be saved with the same
                           name but with a .json extension.
        traj_paths (list): A list of paths to the input trajectory files (H5 files). The corresponding JSON files
                           should have the same name but with a .json extension.
        recompute_id (bool): If True, recompute the episode IDs to ensure they are unique. If False, keep the original
                             episode IDs.
        success (bool): If True, only merge successful trajectories.
    Raises:
        AssertionError: If there is a conflict in the episode IDs when recompute_id is False.
    """
    logger.info(f"Merging {output_path}")
    
    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"episodes": []}
    cnt = 0
    success_cnt = 0
    for traj_path in traj_paths:
        traj_path = str(traj_path)
        logger.info(f"Merging{traj_path}")

        with h5py.File(traj_path, "r") as h5_file:
            json_data = load_json(traj_path.replace(".h5", ".json"))
            
            # For keys other than episodes, keep the first data
            # and check if there is any conflict with other data.
            for key, value in json_data.items():
                if key == "episodes":
                    continue
                if key not in merged_json_data:
                    merged_json_data[key] = value
                else:
                    if merged_json_data[key] != value:
                        logger.warning(f"Conflict detected for key {key} in {traj_path}: {merged_json_data[key]} != {value}")

            # Merge episodes
            for ep in json_data["episodes"]:
                episode_id = ep["episode_id"]
                traj_id = f"traj_{episode_id}"

                if success and not ep.get("success", False):
                    logger.info(f"Skipping unsuccessful trajectory {traj_id} from {traj_path}")
                    continue

                # Copy h5 data
                if recompute_id:
                    new_traj_id = f"traj_{cnt}"
                else:
                    new_traj_id = traj_id

                assert new_traj_id not in merged_h5_file, new_traj_id
                h5_file.copy(traj_id, merged_h5_file, new_traj_id)

                # Copy json data
                if recompute_id:
                    ep["episode_id"] = cnt
                merged_json_data["episodes"].append(ep)

                cnt += 1
                if ep.get("success", False):
                    success_cnt += 1

    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)
    logger.info(f"Merged {cnt} trajectories total, {success_cnt} successful trajectories")

def find_all_subdirs(input_dirs):
    """Return every immediate subdirectory under each input directory, sorted."""
    subdirs = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            # Only descend one level — trial directories live directly under
            # the recording root, not nested deeper.
            for item in input_path.iterdir():
                if item.is_dir():
                    subdirs.append(item)
    return sorted(subdirs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="+")
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="trajectory.h5")
    parser.add_argument("-s", "--success", type=bool, default=True, help="only merge successful trajectories")

    args = parser.parse_args()

    trial_dirs = find_all_subdirs(args.input_dirs)
    print(f"Found {len(trial_dirs)} trial directories:")
    for d in trial_dirs:
        print(f"  {d}")

    traj_paths = []
    for trial_dir in trial_dirs:
        traj_file = trial_dir / args.pattern
        if traj_file.exists():
            traj_paths.append(traj_file)
    
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    merge_trajectories(args.output_path, traj_paths, args.success)

if __name__ == "__main__":
    main()
