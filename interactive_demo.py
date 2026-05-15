#!/usr/bin/env python3
import gymnasium as gym
import mani_skill.envs
import core.env
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from utils import get_grasp_pose_from_config
from scipy.spatial.transform import Rotation as R
import ipdb
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def main():
    """主函数：运行交互式抓取控制程序"""
    print("=== 交互式机械臂抓取控制程序 ===")
    print("正在初始化环境...")

    # 创建环境
    env = gym.make(
        'grasp_part',           # 环境ID
        num_envs=1,             # 环境数量
        obs_mode='state',       # 观察模式
        control_mode='pd_joint_pos',  # 控制模式
        render_mode="human",       # 渲染模式
        object_name='100693',  # 要抓取的物体
        part_name='up_handle'    # 要抓取的部件
    )
    
    # 初始化环境（使用固定位置）
    env.reset()  # 固定位置 (0, 0)，从高空掉落
    # apply_table_texture(env)
    # 创建运动规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=True,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=True,
        print_env_info=False,
    )

    env = env.unwrapped

    # 获取配置和抓取部件信息
    config = env.unwrapped.object_config
    grasp_parts = config.get("grasp_parts", {})

    # 获取指定部件的所有抓取配置
    all_configs = []
    target_part = env.unwrapped.part_name  # 获取环境中的part_name ('cap')
    for part_name_, grasp_list in grasp_parts.items():
        if target_part in part_name_:  # 只包含指定的part_name
            for i in range(len(grasp_list)):
                all_configs.append((part_name_, i))
    # 获取viewer
    # ipdb.set_trace()
    viewer = env.unwrapped.viewer
    # 初始化变量
    current_idx = 0
    current_grasp_pose = None
    current_visual_actor = None

    print("✓ 环境初始化完成")
    print(f"当前part_name: {env.unwrapped.part_name}")
    print(f"发现 {len(all_configs)} 个抓取配置")
    if all_configs:
        print(f"配置示例: {all_configs[:3]}")  # 显示前3个配置
    print("\n=== 按键说明 ===")
    print("  'r' - 显示下一个抓取姿态")
    print("  'g' - 对当前姿态执行抓取")
    print("  'q' - 退出程序")
    print("===============\n")

    try:
        # 主控制循环
        while not viewer.closed:
            # 渲染环境
            
            env.render()

            # 按键控制
            if viewer.window.key_down('r'):  # 显示下一个抓取姿态
                if all_configs:
                    part_name, grasp_id = all_configs[current_idx]
                    print(f"\n显示抓取姿态 {current_idx + 1}/{len(all_configs)}: {part_name}[{grasp_id}]")
                    current_grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)
                    # 使用motion planner内部的可视化
                    if current_grasp_pose is not None:
                        planner._update_grasp_visual(current_grasp_pose)
                        print(f"✓ 显示抓取姿态: {part_name}[{grasp_id}] - 位置 {current_grasp_pose.p}")
                    current_idx = (current_idx + 1) % len(all_configs)
                # ipdb.set_trace()

            elif viewer.window.key_down('t'):  # 重置环境
                print("重置环境...")
                try:
                    # 重置环境
                    obs = env.reset()
                    
                    # 重新初始化运动规划器（如果需要）
                    planner = PandaArmMotionPlanningSolver(
                        env,
                        debug=False,
                        vis=True,
                        base_pose=env.unwrapped.agent.robot.pose,
                        visualize_target_grasp_pose=True,
                        print_env_info=False,
                    )
                    
                    # 重新构建抓取配置（以防环境参数改变）
                    config = env.unwrapped.object_config
                    grasp_parts = config.get("grasp_parts", {})
                    all_configs = []
                    target_part = env.unwrapped.part_name
                    for part_name_, grasp_list in grasp_parts.items():
                        if target_part in part_name_:
                            for i in range(len(grasp_list)):
                                all_configs.append((part_name_, i))

                    # 重置变量
                    current_idx = 0
                    current_grasp_pose = None
                    current_visual_actor = None
                    
                    print("✓ 环境重置完成")
                    
                except Exception as e:
                    print(f"重置环境出错: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif viewer.window.key_down('g'):  # 执行抓取
                # import ipdb
                q_yaw_pi = euler2quat(0, 0, np.pi)  # [w,x,y,z]  roll,pitch,yaw
                q_new = qmult(current_grasp_pose.q, q_yaw_pi)
                tmp_pose = sapien.Pose(current_grasp_pose.p, q_new)
                planner._update_grasp_visual(tmp_pose)
                current_grasp_pose = tmp_pose
                # ipdb.set_trace()

                if current_grasp_pose is not None:
                    print("执行抓取动作...")
                    try:
                        # 获取当前TCP位置
                        current_tcp_pose = env.agent.tcp.pose
                        current_pos = current_tcp_pose.p.cpu().numpy()[0]  # 处理张量

                        # 目标抓取位置
                        target_pos = current_grasp_pose.p

                        print(f"当前TCP位置: {current_pos}")
                        print(f"目标抓取位置: {target_pos}")

                        # 计算合适的预抓取位置
                        # 使用目标位置的x,y，但调整z到合理的高度
                        safe_z = min(target_pos[2], current_pos[2] + 0.1)# 不超过当前高度+20cm
                        safe_z = max(safe_z, 0.3)  # 不低于30cm

                        pre_grasp_pos = np.array([target_pos[0], target_pos[1], safe_z])
                        print(f"调整后预抓取位置: {pre_grasp_pos}")

                        # 创建预抓取姿态（保持抓取姿态的旋转）
                        pre_grasp_pose = sapien.Pose(pre_grasp_pos, current_grasp_pose.q)
                        # pre_grasp_pose = sapien.Pose(pre_grasp_pos, current_tcp_pose.q.cpu().numpy()[0])
                        # 移动到预抓取位置
                        print("移动到预抓取位置...")


                        success = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
                        if success:
                            print("✓ 成功到达预抓取位置")

                            # 计算最终抓取位置（稍微靠近一些）
                            final_grasp_pos = np.array([target_pos[0], target_pos[1], target_pos[2] - 0.02])
                            final_grasp_pose = sapien.Pose(final_grasp_pos, current_grasp_pose.q)
                            
                            # 移动到最终抓取位置
                            print("移动到最终抓取位置...")
                            success = planner.move_to_pose_with_screw(final_grasp_pose)
                            if success:
                                print("✓ 成功到达抓取位置")
                                # 闭合夹爪
                                print("闭合夹爪...")
                                obs, reward, terminated, truncated, info = planner.close_gripper()
                                planner.move_to_pose_with_RRTConnect(sapien.Pose(current_tcp_pose.p.cpu().numpy()[0], current_tcp_pose.q.cpu().numpy()[0]))
                                print(f"抓取完成! 状态: {info.get('is_grasped', 'unknown')}")
                                # planner.open_gripper()

                            else:
                                print("✗ 无法到达最终抓取位置")
                        else:
                            print("✗ 无法到达预抓取位置")


                    except Exception as e:
                        print(f"抓取执行出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("请先按 'r' 显示抓取姿态")

            elif viewer.window.key_down('q'):  # 退出
                print("退出程序...")
                break

            import time
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理环境
        env.close()
        print("程序结束")

if __name__ == "__main__":
    main()