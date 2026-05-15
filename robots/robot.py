from pathlib import Path

from mani_skill.agents.base_agent import BaseAgent, Keyframe  # noqa: F401
from mani_skill.agents.controllers import *  # noqa: F401, F403
from mani_skill.agents.registration import register_agent

# Path computed relative to this file so the URDF is found regardless of where
# the package is checked out. The previous hardcoded /home/robot/workspace/...
# only worked on the original development machine.
_URDF = Path(__file__).resolve().parent / "codroidRobot_description_edu" / "codroidRobot.urdf"


@register_agent()
class CuzRobot(BaseAgent):
    uid = "cuzhuo"
    urdf_path = str(_URDF)
