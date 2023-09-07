import ast
import subprocess
import sys


def test_dont_import_pandas() -> None:
    """Verifies that when we import TileDB, we don't import Pandas eagerly."""
    # Get a list of all modules from a completely fresh interpreter.
    all_mods_str = subprocess.check_output(
        (sys.executable, "-c", "import sys, tiledb; print(list(sys.modules))")
    )
    all_mods = ast.literal_eval(all_mods_str.decode())
    assert "pandas" not in all_mods
