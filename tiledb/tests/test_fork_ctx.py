"""Tests combining fork with tiledb context threads.

Background: the core tiledb library uses threads and it's easy to
experience deadlocks when forking a process that is using tiledb.  The
project doesn't have a solution for this at the moment other than to
avoid using fork(), which is the same recommendation that Python makes.
Python 3.12 warns if you fork() when multiple threads are detected and
Python 3.14 will make it so you never accidentally fork():
multiprocessing will default to "spawn" on Linux.
"""

import subprocess
import sys

import pytest


def run_in_subprocess(code):
    """Runs code in a separate subprocess."""
    script = f"""
import os
import warnings
import tiledb
import multiprocessing

warnings.simplefilter('error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def wrapper_func():
    {code}

wrapper_func()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_no_warning_fork_without_ctx():
    """Get no warning if no tiledb context exists."""
    run_in_subprocess(
        """
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    else:
        os.wait()
    """
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_warning_fork_with_ctx():
    """Get a warning if we fork after creating a tiledb context."""
    run_in_subprocess(
        """
    _ = tiledb.Ctx()
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    else:
        os.wait()
    """
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_warning_fork_with_default_ctx():
    """Get a warning if we fork after creating a default context."""
    run_in_subprocess(
        """
    _ = tiledb.default_ctx()
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    else:
        os.wait()
    """
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_no_warning_multiprocessing_without_ctx():
    """Get no warning if no tiledb context exists."""
    run_in_subprocess(
        """
    mp = multiprocessing.get_context("fork")
    p = mp.Process()
    p.start()
    p.join()
    """
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_warning_multiprocessing_with_ctx():
    """Get a warning if we fork after creating a tiledb context."""
    run_in_subprocess(
        """
    _ = tiledb.Ctx()
    mp = multiprocessing.get_context("fork")
    p = mp.Process()
    p.start()
    p.join()
    """
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_warning_multiprocessing_with_default_ctx():
    """Get a warning if we fork after creating a default context."""
    run_in_subprocess(
        """
    _ = tiledb.default_ctx()
    mp = multiprocessing.get_context("fork")
    p = mp.Process()
    p.start()
    p.join()
    """
    )
