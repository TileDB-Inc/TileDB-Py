"""
Run all test casess
"""

import os
import sys
import unittest


def suite():
    test_dir = os.path.dirname(__file__)
    return unittest.TestLoader().discover(start_dir=test_dir, pattern="test_*.py")


def suite_test():
    """
    suite_test()

    Run all the tests in the test suite
    """

    ret = unittest.TextTestRunner(verbosity=2).run(suite())
    sys.exit(not ret.wasSuccessful())


if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())
