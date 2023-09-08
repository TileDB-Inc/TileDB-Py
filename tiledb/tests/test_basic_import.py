import ast
import subprocess
import sys

import importlib_metadata
from packaging.version import Version


def test_dont_import_pandas() -> None:
    """Verifies that when we import TileDB, we don't import Pandas eagerly."""

    # If tiledb.cloud < 0.10.21 is installed, we should remove it from the module list
    # before running the test, because it eagerly imported pandas before that version.
    # Note that we import tiledb.cloud within tiledb-py, if available, in order to hook
    # Array.apply and other functionality.
    try:
        ver = importlib_metadata.version("tiledb.cloud")
    except importlib_metadata.PackageNotFoundError:
        ver = None
    if ver and Version(ver) < Version("0.10.21"):
        maybe_cloud = "sys.modules['tiledb.cloud'] = None;"
    else:
        maybe_cloud = ""
    # Get a list of all modules from a completely fresh interpreter.
    all_mods_str = subprocess.check_output(
        (
            sys.executable,
            "-c",
            f"import sys; {maybe_cloud} import tiledb; print(list(sys.modules))",
        )
    )
    all_mods = ast.literal_eval(all_mods_str.decode())
    assert "pandas" not in all_mods
