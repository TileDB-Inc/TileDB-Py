import os, glob, unittest, tempfile, sys
import subprocess

def run_checked(args):
  with tempfile.TemporaryDirectory() as tmp:
    cmd = [sys.executable] + args
    proc = subprocess.Popen(cmd, cwd=tmp, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    status = proc.returncode
    
    if status != 0:
      print("Call failed: {}".format(args))
      print("--- stdout:")
      print(out.decode())
      print("--- stderr:")
      print(err.decode())
      sys.exit(1)

class ExamplesTest(unittest.TestCase):

  def test_examples(self):
    examples_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../../examples"))
    for ex in glob.glob(examples_path+"/*.py"):
      args = [ex]
      run_checked(args)

    if sys.version_info >= (3,6):
      doctest_args = ['-m', 'doctest', '-o', 'NORMALIZE_WHITESPACE', '-f',
                      os.path.abspath(os.path.join(__file__, '../../', 'libtiledb.pyx'))]
      run_checked(doctest_args)

