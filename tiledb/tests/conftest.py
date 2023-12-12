import ctypes
import os
import sys

import pytest

import tiledb

from .common import DiskTestCase


# fixture wrapper to use with pytest:
# mark.parametrize does not work with DiskTestCase subclasses
# (unittest.TestCase methods cannot take arguments)
@pytest.fixture(scope="class")
def checked_path():
    dtc = DiskTestCase()
    dtc.setup_method()
    yield dtc
    dtc.teardown_method()


if sys.platform != "win32":

    @pytest.fixture(scope="function", autouse=True)
    def no_output(capfd):
        yield

        # flush stdout
        libc = ctypes.CDLL(None)
        libc.fflush(None)

        out, err = capfd.readouterr()
        if out or err:
            pytest.fail(f"Output captured: {out + err}")


def pytest_addoption(parser):
    parser.addoption("--vfs", default="file")
    parser.addoption("--vfs-config", default=None)


def pytest_configure(config):
    # we need to try importing here so that we don't potentially cause
    # a slowdown in the DenseArray/SparseArray.__new__ path when
    # running `tiledb.open`.
    try:
        import tiledb.cloud  # noqa: F401
    except ImportError:
        pass

    # default must be set here rather than globally
    pytest.tiledb_vfs = "file"

    vfs_config(config)


def vfs_config(pytestconfig):
    vfs_config_override = {}

    vfs = pytestconfig.getoption("vfs")
    if vfs == "s3":
        pytest.tiledb_vfs = "s3"

        vfs_config_override.update(
            {
                "vfs.s3.endpoint_override": "localhost:9999",
                "vfs.s3.aws_access_key_id": "minio",
                "vfs.s3.aws_secret_access_key": "miniosecretkey",
                "vfs.s3.scheme": "https",
                "vfs.s3.verify_ssl": False,
                "vfs.s3.use_virtual_addressing": False,
            }
        )

    vfs_config_arg = pytestconfig.getoption("vfs-config", None)
    if vfs_config_arg:
        pass

    tiledb._orig_ctx = tiledb.Ctx

    def get_config(config):
        final_config = {}
        if isinstance(config, tiledb.Config):
            final_config = config.dict()
        elif config:
            final_config = config

        final_config.update(vfs_config_override)
        return final_config

    class PatchedCtx(tiledb.Ctx):
        def __init__(self, config=None):
            super().__init__(get_config(config))

    class PatchedConfig(tiledb.Config):
        def __init__(self, params=None):
            super().__init__(get_config(params))

    tiledb.Ctx = PatchedCtx
    tiledb.Config = PatchedConfig


@pytest.fixture(scope="function", autouse=True)
def isolate_os_fork(original_os_fork):
    """Guarantee that tests start and finish with no os.fork patch."""
    # Python 3.12 warns about fork() and threads. Tiledb only patches
    # os.fork for Pythons 3.8-3.11.
    if original_os_fork:
        tiledb.ctx._needs_fork_wrapper = True
        os.fork = original_os_fork
    yield
    if original_os_fork:
        tiledb.ctx._needs_fork_wrapper = True
        os.fork = original_os_fork


@pytest.fixture(scope="session")
def original_os_fork():
    """Provides the original unpatched os.fork."""
    if sys.platform != "win32":
        return os.fork
