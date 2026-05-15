from setuptools import setup, find_packages

setup(
    name="diffusion_policy_3d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "zarr==2.12.0",
        # "dm_control",
        "omegaconf",
        "hydra-core==1.2.0",
        "dill==0.3.5.1",
        "einops==0.4.1",
        "diffusers==0.11.1",
        "numba==0.56.4",
        "moviepy",
        "imageio",
        "av",
        "matplotlib",
        "termcolor",
        "ipdb",
        "gpustat",
        "wandb",
        "mani_skill"
    ],
    description="A minimal setup for 3D Diffusion Policy for ManiSkill",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
