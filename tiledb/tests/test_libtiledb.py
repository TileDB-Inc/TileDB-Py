from nose.tools import eq_
from tiledb import libtiledb_version


def test_version():
    eq_(libtiledb_version(), (1, 0, 0))
