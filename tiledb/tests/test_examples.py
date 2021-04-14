import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

run_env = os.environ.copy()
run_env.update({"IN_TEST": "1"})


def run_checked(*args):
    # run example script
    # - in a separate process
    # - in tmpdir so we don't pollute the source tree
    # - with exit status checking (should fail tests if example fails)
    # - also remove the tmp tree, which can catch windows errors

    tmp = tempfile.mkdtemp()
    proc = subprocess.Popen(
        [sys.executable, *args],
        cwd=tmp,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=run_env,
    )
    out, err = proc.communicate()
    status = proc.returncode
    if status != 0:
        print("Call failed: {}".format(args))
        print("--- stdout:")
        print(out.decode())
        print("--- stderr:")
        print(err.decode())
        sys.exit(1)

    shutil.rmtree(tmp)


class ExamplesTest:
    """Test runnability of scripts in examples/"""

    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    @pytest.mark.parametrize(
        "path", glob.glob(os.path.join(PROJECT_DIR, "examples", "*.py"))
    )
    def test_examples(self, path):
        run_checked(path)

    @pytest.mark.skipif(
        sys.platform == "win32" or sys.version_info < (3, 6),
        reason="Some doctests are missing a clean-up step on windows",
    )
    @pytest.mark.parametrize(
        "path", [os.path.join(PROJECT_DIR, "tiledb", "libtiledb.pyx")]
    )
    def test_docs(self, path):
        run_checked("-m", "doctest", "-o", "NORMALIZE_WHITESPACE", "-f", path)
