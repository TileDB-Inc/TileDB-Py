import doctest
import glob
import os
import subprocess
import sys
import tempfile

import pytest

from .common import has_pandas


# override locally to avoid conflict with capsys used below
@pytest.fixture(scope="function", autouse=True)
def no_output():
    pass


class ExamplesTest:
    """Test runnability of scripts in examples/"""

    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    @pytest.mark.parametrize(
        "path", glob.glob(os.path.join(PROJECT_DIR, "examples", "*.py"))
    )
    def test_examples(self, path):
        # run example script
        # - in a separate process
        # - in tmpdir so we don't pollute the source tree
        # - with exit status checking (should fail tests if example fails)
        requires_pd = [
            os.path.join(self.PROJECT_DIR, "examples", f"{fn}.py")
            for fn in [
                "incomplete_iteration",
                "parallel_csv_ingestion",
                "query_condition_datetime",
            ]
        ]
        if not has_pandas() and path in requires_pd:
            pytest.mark.skip("pandas>=1.0,<3.0 not installed")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    subprocess.run(
                        [sys.executable, path],
                        cwd=tmpdir,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding="utf8",
                    )
                except subprocess.CalledProcessError as ex:
                    pytest.fail(ex.stderr, pytrace=False)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Some doctests are missing a clean-up step on windows",
    )
    @pytest.mark.parametrize(
        "path",
        [
            os.path.join(PROJECT_DIR, "tiledb", "libtiledb.pyx"),
            os.path.join(PROJECT_DIR, "tiledb", "fragment.py"),
        ],
    )
    def test_docs(self, path, capsys):
        failures, _ = doctest.testfile(
            path,
            module_relative=False,
            verbose=False,
            optionflags=doctest.NORMALIZE_WHITESPACE,
        )
        if failures:
            stderr = capsys.readouterr().out
            if "No module named 'pandas'" in stderr or (
                "Pandas version >= 1.0 and < 3.0 required for dataframe functionality"
                in stderr
                and not has_pandas()
            ):
                pytest.skip("pandas>=1.0,<3.0 not installed")
            else:
                pytest.fail(stderr)
