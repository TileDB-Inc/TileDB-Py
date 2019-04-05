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

# build python27 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py
git -C TileDB-Py checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py
/opt/python/cp27-cp27mu/bin/python2.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# build python37 wheel
cd /home/tiledb
git clone https://github.com/TileDB-Inc/TileDB-Py TileDB-Py3
git -C TileDB-Py3 checkout $TILEDB_PY_VERSION

cd /home/tiledb/TileDB-Py3
/opt/python/cp37-cp37m/bin/python3.7 setup.py build_ext bdist_wheel --tiledb=/usr/local
auditwheel repair dist/*.whl

# copy build products out
cp /home/tiledb/TileDB-Py/wheelhouse/* /wheels && \
cp /home/tiledb/TileDB-Py3/wheelhouse/* /wheels

