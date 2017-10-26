from nose.tools import eq_
from tiledb import hello_world


def test_hello():
    eq_(hello_world(), "hello world")
