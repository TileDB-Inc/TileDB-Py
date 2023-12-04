"""Tests combining multiprocessing with threads"""

import multiprocessing
import os
import sys
import warnings
from functools import wraps

import pytest


def func():
    return 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_os_warn_at_fork():
    """Get a warning if we fork after importing tiledb."""
    # Background: the core tiledb library uses threads and it's easy to
    # experience deadlocks when forking a process that is using tiledb.
    # The project doesn't have a solution for this at the moment other
    # than to avoid using fork(), which is the same recommendation that
    # Python makes. Python 3.12 warns if you fork() when multiple
    # threads are detected and Python 3.14 will make it so you never
    # accidentally fork(): multiprocessing will default to "spawn" on
    # Linux.

    def wrap_fork():
        os_fork = os.fork

        @wraps(os_fork)
        def wrapper():
            warnings.warn(
                "TileDB is a multithreading library and deadlocks are "
                "likely if fork() is called after a TileDB array has "
                "been created or accessed. "
                "To safely use TileDB with multiprocessing or "
                "concurrent.futures, choose 'spawn' as the start "
                "method for child processes.",
                UserWarning,
            )
            return os_fork()

        return wrapper

    import tiledb  # noqa

    if sys.version_info < (3, 12):
        os.fork = wrap_fork()

    with pytest.warns(UserWarning):
        pid = os.fork()

        if pid == 0:
            func()
            os._exit(0)
        else:
            os.wait()


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork() is not available on Windows"
)
def test_multiprocessing_warn_at_fork():
    """Get a warning if we fork after importing tiledb."""
    # Background: the core tiledb library uses threads and it's easy to
    # experience deadlocks when forking a process that is using tiledb.
    # The project doesn't have a solution for this at the moment other
    # than to avoid using fork(), which is the same recommendation that
    # Python makes. Python 3.12 warns if you fork() when multiple
    # threads are detected and Python 3.14 will make it so you never
    # accidentally fork(): multiprocessing will default to "spawn" on
    # Linux.

    def wrap_fork():
        os_fork = os.fork

        @wraps(os_fork)
        def wrapper():
            warnings.warn(
                "TileDB is a multithreading library and deadlocks are "
                "likely if fork() is called after a TileDB array has "
                "been created or accessed. "
                "To safely use TileDB with multiprocessing or "
                "concurrent.futures, choose 'spawn' as the start "
                "method for child processes.",
                UserWarning,
            )
            return os_fork()

        return wrapper

    import tiledb  # noqa

    if sys.version_info < (3, 12):
        os.fork = wrap_fork()

    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=func)

    with pytest.warns(UserWarning):
        p.start()

    p.join()
