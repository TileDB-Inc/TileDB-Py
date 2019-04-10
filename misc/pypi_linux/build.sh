#!/bin/sh

# USAGE
#------
# 1) docker build .
# - copy resulting IMAGE_HASH
# 2) docker run -v wheels:/wheels -ti IMAGE_HASH build.sh
#
# testing (e.g. using the official python docker images)
# - $ docker run -v `pwd`/wheels:/wheels --rm -ti python bash
# -- pip3 install /wheels/*cp37*.whl
# -- python3.7 -c "import tiledb; print(tiledb.libtiledb.version())"
set -ex

export TILEDB_PY_VERSION=0.4.1

# build python27 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py TileDB-Py27
git -C TileDB-Py27 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py27
# adding -lrt as a work-around for now because python2.7 doesn't link it, but it
# ends up as an unlinked dependency.
CFLAGS="-lrt" /opt/python/cp27-cp27mu/bin/python2.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# build python35 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py TileDB-Py35
git -C TileDB-Py35 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py35
/opt/python/cp35-cp35m/bin/python3.5 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl


# build python36 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py TileDB-Py36
git -C TileDB-Py36 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py36
/opt/python/cp36-cp36m/bin/python3.6 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl


# build python37 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py TileDB-Py37
git -C TileDB-Py37 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py37
/opt/python/cp37-cp37m/bin/python3.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# copy build products out
cp /home/tiledb/TileDB-Py27/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py35/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py36/wheelhouse/* /wheels
cp /home/tiledb/TileDB-Py37/wheelhouse/* /wheels
