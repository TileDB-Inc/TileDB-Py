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


@pytest.fixture
def vfs_config() -> dict[str, str]:
    config: dict[str, str] = {}
    # Configure S3
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        config["vfs.s3.aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
        config["vfs.s3.aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        if os.getenv("VFS_S3_USE_MINIO"):
            config["vfs.s3.endpoint_override"] = "localhost:9999"
            config["vfs.s3.scheme"] = "https"
            config["vfs.s3.use_virtual_addressing"] = "false"
            config["vfs.s3.verify_ssl"] = "false"

    #   Configure Azure
    if os.getenv("AZURE_BLOB_ENDPOINT"):
        config["vfs.azure.blob_endpoint"] = os.getenv("AZURE_BLOB_ENDPOINT")
    if os.getenv("AZURE_STORAGE_ACCOUNT_TOKEN"):
        config["vfs.azure.storage_sas_token"] = os.getenv("AZURE_STORAGE_ACCOUNT_TOKEN")
    elif os.getenv("AZURE_STORAGE_ACCOUNT_NAME") and os.getenv(
        "AZURE_STORAGE_ACCOUNT_KEY"
    ):
        config["vfs.azure.storage_account_name"] = os.getenv(
            "AZURE_STORAGE_ACCOUNT_NAME"
        )
        config["vfs.azure.storage_account_key"] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

    #   Configure Google Cloud
    if os.getenv("TILEDB_TEST_GCS_ENDPOINT"):
        config["vfs.gcs.endpoint"] = os.getenv("TILEDB_TEST_GCS_ENDPOINT")

    return config
