import argparse
import os
import zarr
import numpy as np
from termcolor import cprint
import copy
import sys
sys.path.insert(0, '/home/DMP/ManiSkill/examples/baselines/diffusion_policy_3d')

from load_trajectories import load_demo_dataset
from diffusion_policy_3d.common.utils import downsample_with_fps

TASK_BOUDNS = {
    'GraspCup-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'PickCube-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'default': [-0.7, -1, 0.01, 1, 1, 100],
}

def main(args):
    save_dir = os.path.join(args.zarr_dir, 'maniskill_' + args.env_name + '_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)

    if args.env_name in TASK_BOUDNS:
        x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[args.env_name]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
    min_bound = [x_min, y_min, z_min]
    max_bound = [x_max, y_max, z_max]

    num_demos = args.num_demos
    cprint(f"Number of demos/episodes : {num_demos}", "yellow")

    trajectories = load_demo_dataset(args.hdf5_path, num_traj=num_demos, concat=False)

    assert len(trajectories['observations']) == num_demos
    assert len(trajectories['actions']) == num_demos

    total_count = 0 # total number of elapsed steps
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    demo_idx = 0

    # loop over demos/trajectories
    while demo_idx < num_demos:
        # img_arrays_sub = []
        point_cloud_arrays_sub = []
        # depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0

        L = trajectories['actions'][demo_idx].shape[0]
        assert len(trajectories['observations'][demo_idx]['pointcloud']['xyzw']) == L + 1
        num_episodes = trajectories['actions'][demo_idx].shape[0]

        # loop over episodes for each trajectory
        for episode_idx in range(num_episodes):
            total_count_sub += 1

            # Process pointcloud [6]
            xyzw = trajectories['observations'][demo_idx]['pointcloud']['xyzw'][episode_idx][...]
            rgb = trajectories['observations'][demo_idx]['pointcloud']['rgb'][episode_idx][...]
            filtered_xyzw = xyzw[xyzw[:, -1] == 1]
            filtered_xyz = filtered_xyzw[...,:3]
            filtered_rgb = rgb[xyzw[:, -1] == 1]
            obs_point_cloud = np.concatenate((filtered_xyz, filtered_rgb), axis=1)

            # TODO (done): Crop to remove table/background and only leave the useful point clouds
            use_point_crop = args.use_point_crop
            if use_point_crop:
                if min_bound is not None:
                    mask = np.all(obs_point_cloud[:, :3] > min_bound, axis=1)
                    obs_point_cloud = obs_point_cloud[mask]
                if max_bound is not None:
                    mask = np.all(obs_point_cloud[:, :3] < max_bound, axis=1)
                    obs_point_cloud = obs_point_cloud[mask]

            # Downsample pointclouds
            num_points = 512
            if obs_point_cloud.shape[0] > num_points:
                obs_point_cloud = downsample_with_fps(obs_point_cloud, num_points=num_points)

            # Process action [7]
            action = trajectories['actions'][demo_idx][episode_idx] # [-1,1]

            # Process robot state [29]
            # including joint positions [7+2], velocities [7+2], goal position [3], end-effector pose [7], is_grasped [1]
            observation = trajectories['observations'][demo_idx]
            # if "pointcloud" in observation:
            #     del observation["pointcloud"]
            # if "sensor_param" in observation:
            #     del observation["sensor_param"]
            # if "sensor_data" in observation:
            #     del observation["sensor_data"]

            agent_idx = {
                'qpos': observation['agent']['qpos'][episode_idx],
                'qvel': observation['agent']['qvel'][episode_idx],
            }
            extra_idx = {
                # 'is_grasped': observation['extra']['is_grasped'][episode_idx],
                'tcp_pose': observation['extra']['tcp_pose'][episode_idx],
                # 'goal_pose': observation['extra']['goal_pos'][episode_idx],
            }
            observation_idx = {
                'agent': agent_idx,
                'extra': extra_idx,
            }
            # Manually flatten observation to avoid mani_skill compatibility issues
            # common.flatten_state_dict has issues with numpy arrays
            qpos = observation['agent']['qpos'][episode_idx]
            qvel = observation['agent']['qvel'][episode_idx]
            tcp_pose = observation['extra']['tcp_pose'][episode_idx]
            obs_robot_state = np.concatenate([
                qpos.flatten() if isinstance(qpos, np.ndarray) else np.array(qpos).flatten(),
                qvel.flatten() if isinstance(qvel, np.ndarray) else np.array(qvel).flatten(),
                tcp_pose.flatten() if isinstance(tcp_pose, np.ndarray) else np.array(tcp_pose).flatten(),
            ])

            # img_arrays_sub.append(obs_img)
            point_cloud_arrays_sub.append(obs_point_cloud)
            # depth_arrays_sub.append(obs_depth)
            state_arrays_sub.append(obs_robot_state)
            action_arrays_sub.append(action)

        total_count += total_count_sub
        episode_ends_arrays.append(copy.deepcopy(total_count))  # the index of the last step of the episode
        point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        state_arrays.extend(copy.deepcopy(state_arrays_sub))
        action_arrays.extend(copy.deepcopy(action_arrays_sub))
        cprint('Demo: {} finished.'.format(demo_idx), 'green')
        demo_idx += 1

    # save data
    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True,
                             compressor=compressor)

    cprint(f'-' * 50, 'cyan')
    # print shape
    # cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(
        f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]',
        'green')
    # cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='GraspCup-v1')
    parser.add_argument('--num_demos', type=int, default=150)
    parser.add_argument('--use_point_crop', type=bool, default=True)
    parser.add_argument('--hdf5_path', type=str, default="~/.maniskill/demos/GraspCup-v1/motionplanning/trajectory_cpu.pointcloud.pd_ee_delta_pose.cpu.h5")
    parser.add_argument('--zarr_dir', type=str, default="./data/")

    args = parser.parse_args()
    main(args)
