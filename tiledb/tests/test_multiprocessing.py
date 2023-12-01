"""Tests combining multiprocessing with threads"""

import multiprocessing
import sys

import pytest


def print_hello():
    print("Hello")


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_warn_on_fork():
    """Get a warning if we fork after importing tiledb."""
    # Background: the core tiledb library uses threads and it's easy to
    # experience deadlocks when forking a process that is using tiledb.
    # The project doesn't have a solution for this at the moment other
    # than to avoid using fork(), which is the same recommendation that
    # Python makes. Python 3.12 warns if you fork() when multiple
    # threads are detected and Python 3.14 will make it so you never
    # accidentally fork(): multiprocessing will default to "spawn" on
    # Linux.

    import tiledb  # noqa

    ctx = multiprocessing.get_context("fork")
    with pytest.warns(DeprecationWarning):
        p = ctx.Process(target=print_hello)
        p.start()
        p.join()
