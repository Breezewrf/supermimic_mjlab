"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.8.1",
    "warp-lang==1.12.0",
    "scipy==1.17.0"
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(include=("src", "src.*")),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
