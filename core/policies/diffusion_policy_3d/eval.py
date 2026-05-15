if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    git_root = str(pathlib.Path(__file__).resolve().parents[3])
    if git_root not in sys.path:
        sys.path.append(git_root)
    try:
        import core.env  # noqa: F401
    except Exception as e:
        print(f"[Warn] Could not import core.env for env registration: {e}")
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()


if __name__ == "__main__":
    main()