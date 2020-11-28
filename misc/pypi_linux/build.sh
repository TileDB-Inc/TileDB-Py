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

# build python27 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py27
git -C TileDB-Py27 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py27
/opt/python/cp27-cp27mu/bin/python2.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp27-cp27mu/bin/python2.7 -m pip install wheelhouse/*.whl
cd tiledb/tests
#/opt/python/cp27-cp27mu/bin/python2.7 -m unittest

# build python35 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py35
git -C TileDB-Py35 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py35
/opt/python/cp35-cp35m/bin/python3.5 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp35-cp35m/bin/python3.5 -m pip install wheelhouse/*.whl
cd tiledb/tests
#/opt/python/cp35-cp35m/bin/python3.5 -m unittest


# build python36 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py36
git -C TileDB-Py36 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py36
/opt/python/cp36-cp36m/bin/python3.6 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp36-cp36m/bin/python3.6 -m pip install wheelhouse/*.whl
cd tiledb/tests
#/opt/python/cp36-cp36m/bin/python3.6 -m unittest


# build python37 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py37
git -C TileDB-Py37 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py37
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
/opt/python/cp38-cp38/bin/python3.8 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp38-cp38/bin/python3.8 -m pip install wheelhouse/*.whl
cd tiledb/tests

# build python39 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py39
git -C TileDB-Py39 checkout $TILEDBPY_VERSION

cd /home/tiledb/TileDB-Py39
/opt/python/cp39-cp39/bin/python3.9 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl
/opt/python/cp39-cp39/bin/python3.9 -m pip install wheelhouse/*.whl

cd tiledb/tests

# copy build products out
cp /home/tiledb/TileDB-Py27/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py35/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py36/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py37/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py38/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py39/wheelhouse/* /wheels
