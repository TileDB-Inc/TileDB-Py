"""
Deprecated, please use `pip install .` instead
"""

from scikit_build_core.setuptools.wrapper import setup

setup(
    cmake_source_dir=".",
    cmake_args=[],
)
