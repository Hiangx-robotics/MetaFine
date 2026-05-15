# 导入机器人定义文件来注册机器人
import robot  # 这会注册你的CuzRobot

# 导入ManiSkill相关模块
from mani_skill.agents import REGISTERED_AGENTS
import mani_skill.examples.demo_robot as demo_robot_script
from mani_skill.examples.demo_robot import Args  # 导入Args类

# 检查机器人是否注册成功
print("已注册的机器人:", list(REGISTERED_AGENTS.keys()))

# 创建Args对象，指定使用你的机器人
args = Args(
    robot_uid="cuzhuo",  # 使用你的机器人
    # render_mode="human"   # 启用可视化
)

# 使用你的机器人运行demo
demo_robot_script.main(args)