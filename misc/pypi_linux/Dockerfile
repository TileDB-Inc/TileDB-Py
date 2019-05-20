FROM quay.io/pypa/manylinux1_x86_64

###############################################
# settings
# NOTE: MUST USE the 'mu' variant here to be compatible
#       with ~most~ linux distros (see manylinux README)
ENV PYTHON_BASE /opt/python/cp27-cp27mu/bin/
ENV TILEDB_VERSION 1.5.0
ENV TILEDB_PY_VERSION 0.4.0

RUN useradd tiledb
ENV HOME /home/tiledb

# dependencies:
# - cmake (need recent) and auditwheel from pip
# - perl 5.10.0 for openssl
RUN  $PYTHON_BASE/pip install cmake auditwheel && \
  curl -L https://install.perlbrew.pl | bash && \
  source $HOME/perl5/perlbrew/etc/bashrc && \
  perlbrew --notest install perl-5.10.0

ENV CMAKE /opt/python/cp27-cp27mu/bin/cmake
# build libtiledb (core)
# notes:
#    1) we are using auditwheel from https://github.com/pypa/auditwheel
#       this verifies and tags wheel products with the manylinux1 label,
#       and allows us to build libtiledb once, install it to a normal
#       system path, and then use it to build wheels for all of the python
#       versions.
#    2) perl-5.10.0, buit above, is required to build OpenSSL
RUN cd /home/tiledb/ && \
  source $HOME/perl5/perlbrew/etc/bashrc && \
  perlbrew use perl-5.10.0 && \
  git clone https://github.com/TileDB-Inc/TileDB && \
  git -C TileDB checkout $TILEDB_VERSION && \
  mkdir build && \
  cd build && \
  $CMAKE -DTILEDB_S3=ON -DTILEDB_HDFS=ON -DTILEDB_TESTS=OFF \
         -DTILEDB_FORCE_ALL_DEPS:BOOL=ON -DSANITIZER="OFF;-DCOMPILER_SUPPORTS_AVX2:BOOL=FALSE" \
         ../TileDB && \
  make -j2 && \
  make install-tiledb

ADD build.sh /usr/bin/build.sh
RUN chmod +x /usr/bin/build.sh
