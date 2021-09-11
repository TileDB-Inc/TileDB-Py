#!/bin/sh

# Usage:
# 1) update version information
# 2) run from root directory of TileDB-Py checkout
# 3) test and upload wheels to PyPI

set -xeu

export LIBTILEDB_VERSION=2.4.0
export TILEDBPY_VERSION=0.10.0

export CIBW_MANYLINUX_AARCH64_IMAGE=wheel-host-aarch64.manylinux2014
export CIBW_SKIP='cp27-* cp35-* cp36-* cp310-* pp-* *_i686 pp*'
export CIBW_PLATFORM='linux'
export CIBW_ENVIRONMENT='TILEDB_PATH=/usr/local/'
export CIBW_BUILD_VERBOSITY=1
export CIBW_BEFORE_TEST="pip install -r misc/requirements_wheel.txt"
export CIBW_TEST_COMMAND="python -c 'import tiledb'"
export TILEDB_WHEEL_BUILD=1

docker build -t $CIBW_MANYLINUX_AARCH64_IMAGE -f misc/pypi_linux/Dockerfile.aarch64.manylinux2014 .

python3 -m venv cibuildwheel_venv
. cibuildwheel_venv/bin/activate
pip install cibuildwheel

cibuildwheel --platform=linux --output-dir=wheelhouse .
