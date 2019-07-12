#!/bin/sh

# USAGE
#------
# 0) cd TileDB-Py (NOTE: root directory!)
# 1) docker build -f misc/pypi_linux/Dockerfile .
# - copy resulting IMAGE_HASH
# 2) docker run -v misc/pypi_linux/wheels:/wheels -ti IMAGE_HASH build.sh
#
# testing (e.g. using the official python docker images)
# - $ docker run -v `pwd`/wheels:/wheels --rm -ti python bash
# -- pip3 install /wheels/*cp37*.whl
# -- python3.7 -c "import tiledb; print(tiledb.libtiledb.version())"
set -ex

export TILEDB_PY_REPO="/opt/TileDB-Py"

# build python27 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py27
git -C TileDB-Py27 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py27
/opt/python/cp27-cp27mu/bin/python2.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# build python35 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py35
git -C TileDB-Py35 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py35
/opt/python/cp35-cp35m/bin/python3.5 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl


# build python36 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py36
git -C TileDB-Py36 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py36
/opt/python/cp36-cp36m/bin/python3.6 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl


# build python37 wheel
cd /home/tiledb
git clone $TILEDB_PY_REPO TileDB-Py37
git -C TileDB-Py37 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py37
/opt/python/cp37-cp37m/bin/python3.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# copy build products out
cp /home/tiledb/TileDB-Py27/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py35/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py36/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py37/wheelhouse/* /wheels
