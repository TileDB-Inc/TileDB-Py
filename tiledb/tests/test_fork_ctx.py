"""Tests combining fork with tiledb context threads.

Background: the core tiledb library uses threads and it's easy to
experience deadlocks when forking a process that is using tiledb.  The
project doesn't have a solution for this at the moment other than to
avoid using fork(), which is the same recommendation that Python makes.
Python 3.12 warns if you fork() when multiple threads are detected and
Python 3.14 will make it so you never accidentally fork():
multiprocessing will default to "spawn" on Linux.
"""

import multiprocessing
import os
import sys
import warnings

import pytest

import tiledb


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_no_warning_fork_without_ctx():
    """Get no warning if no tiledb context exists."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        else:
            os.wait()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_warning_fork_with_ctx():
    """Get a warning if we fork after creating a tiledb context."""
    _ = tiledb.Ctx()
    with pytest.warns(UserWarning, match="TileDB is a multithreading library"):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        else:
            os.wait()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_warning_fork_with_default_ctx():
    """Get a warning if we fork after creating a default context."""
    _ = tiledb.default_ctx()
    with pytest.warns(UserWarning, match="TileDB is a multithreading library"):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        else:
            os.wait()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_no_warning_multiprocessing_without_ctx():
    """Get no warning if no tiledb context exists."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mp = multiprocessing.get_context("fork")
        p = mp.Process()
        p.start()
        p.join()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_warning_multiprocessing_with_ctx():
    """Get a warning if we fork after creating a tiledb context."""
    _ = tiledb.Ctx()
    mp = multiprocessing.get_context("fork")
    p = mp.Process()
    with pytest.warns(UserWarning, match="TileDB is a multithreading library"):
        p.start()
    p.join()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_warning_multiprocessing_with_default_ctx():
    """Get a warning if we fork after creating a default context."""
    _ = tiledb.default_ctx()
    mp = multiprocessing.get_context("fork")
    p = mp.Process()
    with pytest.warns(UserWarning, match="TileDB is a multithreading library"):
        p.start()
    p.join()
