import os, sys, glob, unittest, tempfile, shutil, platform
import subprocess

run_env = os.environ.copy()
run_env.update({"IN_TEST": "1"})


def run_checked(args):
    # run example script
    # - in a separate process
    # - in tmpdir so we don't pollute the source tree
    # - with exit status checking (should fail tests if example fails)
    # - also remove the tmp tree, which can catch windows errors

    tmp = tempfile.mkdtemp()
    cmd = [sys.executable] + args
    proc = subprocess.Popen(
        cmd, cwd=tmp, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=run_env
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


class ExamplesTest(unittest.TestCase):
    """Test runnability of scripts in examples/"""

    def test_examples(self):
        # construct the abspath to the examples directory
        examples_path = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], "../../examples")
        )
        for ex in glob.glob(examples_path + "/*.py"):
            args = [ex]
            run_checked(args)

    # TODO some of the doctests are missing a clean-up step on windows
    @unittest.skipIf(platform.system() == "Windows", "")
    def test_docs(self):
        if sys.version_info >= (3, 6):
            doctest_args = [
                "-m",
                "doctest",
                "-o",
                "NORMALIZE_WHITESPACE",
                "-f",
                os.path.abspath(os.path.join(__file__, "../../", "libtiledb.pyx")),
            ]
            run_checked(doctest_args)
