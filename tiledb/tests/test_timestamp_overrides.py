import datetime
import os
import subprocess
import sys

import numpy as np
import pytest

import tiledb
from tiledb.main import PyFragmentInfo
from tiledb.tests.common import DiskTestCase


def has_libfaketime():
    try:
        subprocess.check_output(["which", "faketime"])
        return True
    except subprocess.CalledProcessError:
        return False


@pytest.mark.skipif(
    sys.platform == "win32" or not has_libfaketime(),
    reason=f"libfaketime not installed. {'Not supported on Windows.' if sys.platform == 'win32' else ''}",
)
class TestTimestampOverrides(DiskTestCase):
    def test_timestamp_overrides(self):
        uri_fragments = self.path("time_test_fragments")
        uri_group_metadata = self.path("time_test_group_metadata")

        python_exe = sys.executable
        cmd = (
            f"from tiledb.tests.test_timestamp_overrides import TestTimestampOverrides; "
            f"TestTimestampOverrides().helper_fragments('{uri_fragments}'); "
            f"TestTimestampOverrides().helper_group_metadata('{uri_group_metadata}')"
        )
        test_path = os.path.dirname(os.path.abspath(__file__))

        try:
            # "+x0" is the time multiplier, which makes the time freeze during the test
            subprocess.check_output(
                ["faketime", "-f", "+x0", python_exe, "-c", cmd], cwd=test_path
            )
        except subprocess.CalledProcessError as e:
            raise e

    def helper_fragments(self, uri):
        start_datetime = datetime.datetime.now()

        fragments = 5
        A = np.zeros(fragments)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 4), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        uris_seen = set()
        chronological_order = []

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w") as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

            # Read the data back immediately after writing to ensure it is correct
            with tiledb.DenseArray(uri, mode="r") as T:
                read_data = T[fragment_idx : fragment_idx + 1]
            self.assertEqual(read_data, np.array([fragment_idx]))

            fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
            uris = fragment_info.get_uri()
            new_uri = set(uris) - uris_seen
            uris_seen.update(uris)
            chronological_order.extend(new_uri)

        end_datetime = datetime.datetime.now()
        self.assertEqual(start_datetime, end_datetime)

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
        final_uris = fragment_info.get_uri()

        # Keep only the last part of the uris
        final_uris = [os.path.basename(uri) for uri in final_uris]
        chronological_order = [os.path.basename(uri) for uri in chronological_order]

        # Check that timestamps are the same (faketime is working)
        timestamps = set()
        for uri in final_uris:
            parts = uri.split("_")
            timestamps.add((parts[2], parts[3]))

        self.assertEqual(len(timestamps), 1)

        # Check that UUIDs are unique
        uuids = set()
        for uri in final_uris:
            parts = uri.split("_")
            uuids.add(parts[4])

        self.assertEqual(len(uuids), fragments)

        # Ensure that write order is correct
        self.assertEqual(chronological_order, sorted(final_uris))

    def helper_group_metadata(self, uri):
        vfs = tiledb.VFS()

        start_datetime = datetime.datetime.now()

        tiledb.Group.create(uri)
        loop_count = 10
        uris_seen = set()
        chronological_order = []
        meta_path = f"{uri}/__meta"

        for i in range(loop_count):
            with tiledb.Group(uri, "w") as grp:
                grp.meta["meta"] = i

            # Read the data back immediately after writing to ensure it is correct
            with tiledb.Group(uri, "r") as grp:
                self.assertEqual(grp.meta["meta"], i)

            uris = vfs.ls(meta_path)
            new_uri = set(uris) - uris_seen
            uris_seen.update(uris)
            chronological_order.extend(new_uri)

        end_datetime = datetime.datetime.now()
        self.assertEqual(start_datetime, end_datetime)

        final_uris = vfs.ls(meta_path)

        # Keep only the last part of the uris
        final_uris = [os.path.basename(uri) for uri in final_uris]
        chronological_order = [os.path.basename(uri) for uri in chronological_order]

        # Check that timestamps are the same (faketime is working)
        timestamps = set()
        for uri in final_uris:
            parts = uri.split("_")
            timestamps.add((parts[2], parts[3]))

        self.assertEqual(len(timestamps), 1)

        # Check that UUIDs are unique
        uuids = set()
        for uri in final_uris:
            parts = uri.split("_")
            uuids.add(parts[4])

        self.assertEqual(len(uuids), loop_count)

        # Ensure that write order is correct
        self.assertEqual(chronological_order, sorted(final_uris))
