#!/bin/sh

# USAGE
#------
# 0) cd TileDB-Py (NOTE: root directory!)
# 1) docker build -f misc/pypi_linux/Dockerfile . -t wheel_builder
# 2) docker run -v `pwd`/misc/pypi_linux/wheels:/wheels -ti wheel_builder build.sh
#
# testing (e.g. using the official python docker images)
# - $ docker run -v `pwd`/misc/pypi_linux/wheels:/wheels --rm -ti python bash
# -- pip3 install /wheels/*cp37*.whl
# -- python3.7 -c "import tiledb; print(tiledb.libtiledb.version()) and assert tiledb.VFS().supports('s3')"
set -ex

export TILEDB_PY_REPO="/opt/TileDB-Py"

# build python37 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py37
git -C TileDB-Py37 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py37
/opt/python/cp37-cp37m/bin/python3.7 -m pip install -r misc/requirements_wheel.txt
/opt/python/cp37-cp37m/bin/python3.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp37-cp37m/bin/python3.7 -m pip install wheelhouse/*.whl
cd tiledb/tests
#/opt/python/cp37-cp37m/bin/python3.7 -m unittest

# build python38 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py38
git -C TileDB-Py38 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py38
/opt/python/cp38-cp38m/bin/python3.8 -m pip install -r misc/requirements_wheel.txt
/opt/python/cp38-cp38/bin/python3.8 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp38-cp38/bin/python3.8 -m pip install wheelhouse/*.whl
cd tiledb/tests

# build python39 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py39
git -C TileDB-Py39 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py39
/opt/python/cp39-cp39m/bin/python3.9 -m pip install -r misc/requirements_wheel.txt
/opt/python/cp39-cp39/bin/python3.9 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp39-cp39/bin/python3.9 -m pip install wheelhouse/*.whl
cd tiledb/tests

# build python310 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py310
git -C TileDB-Py310 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py310
/opt/python/cp310-cp310m/bin/python3.10 -m pip install -r misc/requirements_wheel.txt
/opt/python/cp310-cp310/bin/python3.10 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp310-cp310/bin/python3.10 -m pip install wheelhouse/*.whl
cd tiledb/tests

# build python311 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py311
git -C TileDB-Py311 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py311
/opt/python/cp311-cp311m/bin/python3.11 -m pip install -r misc/requirements_wheel.txt
/opt/python/cp311-cp311/bin/python3.11 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp311-cp311/bin/python3.11 -m pip install wheelhouse/*.whl
cd tiledb/tests

# copy build products out
cp /home/tiledb/TileDB-Py37/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py38/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py39/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py310/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py311/wheelhouse/* /wheels
