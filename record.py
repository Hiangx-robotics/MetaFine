import os
import os.path as osp
import shutil
import gymnasium as gym
import core.env
import core.skill
import numpy as np
import sapien.core as sapien
import multiprocessing as mp
import argparse
import time
from random import randint
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers.record import RecordEpisode 
from tqdm import tqdm
from core.common import MP_SOLUTIONS
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="peg_in_hole", help=f"Environment to run motion planning solver on. Available options are")
    parser.add_argument(
        "--skill-type",
        type=str,
        default=None,
        choices=list(MP_SOLUTIONS.keys()),
        help="Skill type to execute. Defaults to env-id when omitted.",
    )
    parser.add_argument("--part-name", type=str, default="button",
                        help="Part name to execute (validated against assets/<object>/grasp_poses.json).")
    parser.add_argument("--object-name", type=str, default="100937",
                        help="Asset folder name under assets/ (validated against assets/<object>/grasp_poses.json).")
    parser.add_argument("--verbose", default=False, action="store_true", help="whether or not to print verbose information")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=5, help="Number of trajectories to generate.")
    parser.add_argument("-vlm", default=False, help="whether or not to use vlm to evaluate the success of the trajectory.")
    
    parser.add_argument("--only-count-success", default=False, action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", default=True, action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, default="trajectory", help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--fox-link", type=str, default=False, help="where to save the recorded trajectories")
    parser.add_argument("--target-letter", type=str, default=None,
        help="Target letter for assembling_kits_metafine_letter (M/E/T/A/F/I/N). Default I.")
    parser.add_argument("--task-graph", type=str, default=None,
        help="Path to a YAML/JSON task graph. When set, the env, object, part and skill chain are taken from the file; --env-id/--skill-type/--object-name/--part-name are ignored. See utils/task_graph.py for the schema.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    return parser.parse_args(args)

def main(args):
    # --- Task-graph mode: load a YAML/JSON chain and run multi_skill env ---
    task_graph = None
    goal_predicate = None
    if args.task_graph is not None:
        from utils.task_graph import load_task_graph
        from core.predicates import compile_predicate
        task_graph = load_task_graph(args.task_graph)
        env_id = task_graph.env
        object_name = task_graph.object or args.object_name
        part_name = task_graph.part or args.part_name
        # The task-graph step list IS the "skill chain"; we no longer pick a
        # single skill_type for the planner — run_task_graph iterates them.
        skill_type = task_graph.steps[0].skill if task_graph.steps else "multi_skill"
        if task_graph.success is not None:
            goal_predicate = compile_predicate(task_graph.success)
    else:
        env_id = args.env_id
        skill_type = args.skill_type or env_id
        if skill_type not in MP_SOLUTIONS:
            available_skills = ", ".join(sorted(MP_SOLUTIONS.keys()))
            raise ValueError(
                f"Unknown skill_type='{skill_type}'. "
                f"Please pass --skill-type explicitly. Available: {available_skills}"
            )
        object_name = args.object_name
        part_name = args.part_name
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name
    solve = MP_SOLUTIONS.get(skill_type)  # may be None in task-graph mode
    successes = []
    failed_motion_plans = 0
    solution_episode_lengths = []
    total_trials = 0
    passed = 0
    pbar = tqdm(range(args.num_traj), desc=f"Skill: {skill_type} Object: {object_name} Part: {part_name}")
    while passed < args.num_traj:
        trial_output_dir = osp.join("./", args.record_dir, f"{skill_type}", f"trial_{total_trials+1:04d}")
        os.makedirs(trial_output_dir, exist_ok=True)
        env = None
        success = torch.tensor([False])

        try:
            extra_env_kwargs = {}
            if args.target_letter is not None:
                extra_env_kwargs["target_letter"] = args.target_letter
            # Envs in the GraspPartEnv family take object_name / part_name to
            # pick which articulated asset and part to load. Bespoke envs
            # (peg_in_hole, plug_charger, stack_pyramid, draw_triangle,
            # assembling_kits, put_blocks_*) construct their scene from
            # fixed primitives and reject these kwargs, so we only pass
            # them when the chosen env can consume them.
            ARTICULATED_ENVS = {
                "grasp_part", "align_to_part", "stand_up",
                "toggle_switch", "toggle_switch_table", "lid_opening",
                "slide_along", "multi_skill", "rotate", "door_env",
                "take_out_and_grasp_part_into_box",
            }
            if env_id in ARTICULATED_ENVS:
                extra_env_kwargs["object_name"] = object_name
                extra_env_kwargs["part_name"] = part_name
            # In task-graph mode the env is always multi_skill, and we feed
            # it the parsed chain plus an optional compiled goal predicate.
            # goal_predicate is a callable, which RecordEpisode would try to
            # JSON-serialize for replay — set it on the unwrapped env after
            # construction instead.
            if task_graph is not None:
                extra_env_kwargs["skill_chain"] = task_graph.to_chain()
            env = gym.make(
                env_id,
                obs_mode=args.obs_mode,
                control_mode="pd_joint_pos",
                render_mode=args.render_mode,
                sensor_configs=dict(shader_pack=args.shader, width=224, height=224),
                human_render_camera_configs=dict(shader_pack=args.shader),
                viewer_camera_configs=dict(shader_pack=args.shader),
                sim_backend=args.sim_backend,
                **extra_env_kwargs,
            )

            env = RecordEpisode(
                env,
                output_dir=trial_output_dir,
                trajectory_name="trajectory",
                save_video=args.save_video,
                record_reward=False,
                video_fps=30,
                save_on_reset=False,
            )
            # Attach the compiled goal predicate after RecordEpisode wraps the
            # env — see comment above for why this isn't a gym.make kwarg.
            if task_graph is not None and goal_predicate is not None:
                env.unwrapped.goal_predicate = goal_predicate
            seed = randint(0, 1000000)
            env.reset(seed=seed)

            planner = PandaArmMotionPlanningSolver(
                env,
                vis=args.vis,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=args.vis,
                print_env_info=False,
            )

            if task_graph is not None:
                from utils.task_graph import run_task_graph
                chain_ok = run_task_graph(planner, task_graph.to_chain(), verbose=args.verbose)
                if not chain_ok:
                    result = -1
                else:
                    # Use the env's own evaluate (which incorporates goal_predicate
                    # if set) to determine final success rather than a per-skill
                    # return value.
                    info = env.unwrapped.evaluate()
                    info.setdefault("elapsed_steps", torch.tensor(0))
                    result = (None, 0.0, False, False, info)
            else:
                result = solve(planner, part_name=args.part_name, verbose=args.verbose)

            if result == -1:
                success = torch.tensor([False])
                failed_motion_plans += 1
            else:
                # result is (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                success = info.get("success", False)
                if success and "elapsed_steps" in info:
                    try:
                        solution_episode_lengths.append(info["elapsed_steps"].item())
                    except AttributeError:
                        solution_episode_lengths.append(int(info["elapsed_steps"]))
    
        except Exception as e:
            print(f"Exception in trial {total_trials+1}: {e}")
            success = torch.tensor([False])
            failed_motion_plans += 1

        successes.append(success)
        total_trials += 1

        # 实时计算统计信息
        current_success_rate = np.mean(successes) if successes else 0.0
        failed_rate = failed_motion_plans / total_trials if total_trials > 0 else 0.0
        avg_length = np.mean(solution_episode_lengths) if solution_episode_lengths else 0.0
        max_length = np.max(solution_episode_lengths) if solution_episode_lengths else 0.0

        if success or not args.only_count_success:
            if env is not None:
                env.flush_trajectory()
                if args.save_video:
                    env.flush_video()
                    if args.verbose:
                        print(f"📹 Saved video to {trial_output_dir}/trajectory.mp4")

            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=f"{current_success_rate:.3f}",
                    trials=f"{passed+1}/{args.num_traj}"
                )
            )
            passed += 1
        else:
            # Failed trial and only_count_success=True, update progress bar
            pbar.set_postfix(
                dict(
                    success_rate=f"{current_success_rate:.3f}",
                    trials=f"{passed}/{args.num_traj}",
                    status="failed"
                )
            )
            if env is not None:
                env.flush_trajectory(save=False)
                if args.save_video:
                    env.flush_video(save=False)
        if env is not None:
            try:
                env.close()
            except Exception as e3:
                print(f"Error closing env: {e3}")

    print(f"\n=== Final Results ===")
    final_success_rate = np.mean(successes) if successes else 0.0
    print(f"Total trials: {total_trials}")
    print(f"Successful trials: {np.sum(successes)}")
    print(f"Success rate: {final_success_rate:.3f}")
    print(f"Failed motion plans: {failed_motion_plans}")
    if solution_episode_lengths:
        print(f"Average episode length: {np.mean(solution_episode_lengths):.1f}")
        print(f"Max episode length: {np.max(solution_episode_lengths)}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    main(args)