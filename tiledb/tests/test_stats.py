import numpy as np
from numpy.testing import assert_array_equal

import tiledb

from .common import (
    DiskTestCase,
    assert_captured,
)


class StatsTest(DiskTestCase):
    def test_stats(self, capfd):
        tiledb.stats_enable()
        tiledb.stats_reset()
        tiledb.stats_disable()

        tiledb.stats_enable()

        path = self.path("test_stats")

        with tiledb.from_numpy(path, np.arange(10)) as T:
            pass

        # check that Writer stats are printed
        tiledb.stats_dump()

        if tiledb.libtiledb.version() >= (2, 27):
            assert_captured(capfd, "Context.Query.Writer")
        else:
            assert_captured(capfd, "Context.StorageManager.Query.Writer")

        # check that Writer stats are not printed because of reset
        tiledb.stats_reset()
        tiledb.stats_dump()

        if tiledb.libtiledb.version() >= (2, 27):
            assert_captured(capfd, "Context.Query.Writer", expected=False)
        else:
            assert_captured(
                capfd, "Context.StorageManager.Query.Writer", expected=False
            )

        with tiledb.open(path) as T:
            tiledb.stats_enable()
            assert_array_equal(T, np.arange(10))

            # test stdout version
            tiledb.stats_dump()
            assert_captured(capfd, "TileDB Embedded Version:")

            # check that Reader stats are printed
            tiledb.stats_dump()
            if tiledb.libtiledb.version() >= (2, 27):
                assert_captured(capfd, "Context.Query.Reader")
            else:
                assert_captured(capfd, "Context.StorageManager.Query.Reader")

            # test string version
            stats_v = tiledb.stats_dump(print_out=False)
            if tiledb.libtiledb.version() < (2, 3):
                self.assertTrue("==== READ ====" in stats_v)
            else:
                self.assertTrue('"timers": {' in stats_v)
            self.assertTrue("==== Python Stats ====" in stats_v)

            stats_quiet = tiledb.stats_dump(print_out=False, verbose=False)
            if tiledb.libtiledb.version() < (2, 3):
                self.assertTrue("Time to load array schema" not in stats_quiet)

                # TODO seems to be a regression, no JSON
                stats_json = tiledb.stats_dump(json=True)
                self.assertTrue(isinstance(stats_json, dict))
                self.assertTrue("CONSOLIDATE_COPY_ARRAY" in stats_json)
            else:
                self.assertTrue("==== READ ====" in stats_quiet)

            # check that Writer stats are not printed because of reset
            tiledb.stats_reset()
            tiledb.stats_dump()
            if tiledb.libtiledb.version() >= (2, 27):
                assert_captured(capfd, "Context.Query.Reader", expected=False)
            else:
                assert_captured(
                    capfd, "Context.StorageManager.Query.Reader", expected=False
                )

    def test_stats_include_python_json(self):
        tiledb.stats_enable()

        path = self.path("test_stats")

        with tiledb.from_numpy(path, np.arange(10)) as T:
            pass

        tiledb.stats_reset()
        with tiledb.open(path) as T:
            tiledb.stats_enable()
            assert_array_equal(T, np.arange(10))
            json_stats = tiledb.stats_dump(print_out=False, json=True)
            assert isinstance(json_stats, str)
            assert "python" in json_stats
            assert "timers" in json_stats
            assert "counters" in json_stats
