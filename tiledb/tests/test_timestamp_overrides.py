import datetime
import os
import subprocess
import sys

import numpy as np
import pytest

import tiledb
from tiledb.main import PyFragmentInfo

from .common import DiskTestCase

# def has_libfaketime():
#     find a way to check if libfaketime is installed


class TimestampOverridesTest(DiskTestCase):
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="libfaketime is not supported on Windows",
    )
    # @pytest.mark.skipif(
    #     not has_libfaketime(),
    #     reason="libfaketime not installed",
    # )
    def test_timestamp_overrides(self):
        uri = self.path("time_test")

        python_exe = sys.executable
        cmd = (
            f"from tiledb.tests.test_timestamp_overrides import TimestampOverridesTest; "
            f"TimestampOverridesTest().helper('{uri}')"
        )
        test_path = os.path.dirname(os.path.abspath(__file__))

        try:
            # "+x0" is the time multiplier, which makes the time freeze during the test
            subprocess.run(
                ["faketime", "-f", "+x0", python_exe, "-c", cmd], cwd=test_path
            )
        except subprocess.CalledProcessError as e:
            raise e

    def helper(self, uri):
        start_datetime = datetime.datetime.now()

        fragments = 25
        A = np.zeros(fragments)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 24), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        uris_seen = set()
        chronological_order = []

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w") as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

            fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
            uris = fragment_info.get_uri()
            new_uris = set(uris) - uris_seen
            uris_seen.update(uris)
            chronological_order.extend(new_uris)

        end_datetime = datetime.datetime.now()
        self.assertTrue(start_datetime == end_datetime)

        # check if fragment_info.get_uri() returns the uris in chronological order
        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
        final_uris = fragment_info.get_uri()
        for uri1, uri2 in zip(chronological_order, final_uris):
            assert uri1 == uri2
