import ast
import subprocess
import sys

from packaging.version import Version


def tiledb_cloud_eagerly_imports_pandas() -> bool:
    try:
        import pandas

        import tiledb.cloud
    except ImportError:
        # Can't import something that's not installed.
        return False
    if Version(tiledb.cloud.__version__) < Version("0.10.21"):
        # Old versions of tiledb-cloud will import Pandas eagerly.
        return True
    if Version(pandas.__version__) < Version("1.5"):
        # If an old version of Pandas is installed, tiledb-cloud needs to
        # import it eagerly to patch it.
        return True
    return False


def test_dont_import_pandas() -> None:
    """Verifies that when we import TileDB, we don't import Pandas eagerly."""

    # We import tiledb.cloud within tiledb-py, if available, in order to hook
    # Array.apply and other functionality.  If the version of tiledb-cloud
    # we have installed would import Pandas eagerly on its own, we need to
    # suppress its importation.
    suppress_cloud = (
        "sys.modules['tiledb.cloud'] = None;"
        if tiledb_cloud_eagerly_imports_pandas()
        else ""
    )
    # Get a list of all modules from a completely fresh interpreter.
    all_mods_str = subprocess.check_output(
        (
            sys.executable,
            "-c",
            f"import sys; {suppress_cloud} import tiledb; print(list(sys.modules))",
        )
    )
    all_mods = ast.literal_eval(all_mods_str.decode())
    assert "pandas" not in all_mods
