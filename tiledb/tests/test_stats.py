import numpy as np
from numpy.testing import assert_array_equal

import tiledb

from .common import (
    DiskTestCase,
    assert_captured,
)


class StatsTest(DiskTestCase):
    def test_stats(self, capfd):
        tiledb.libtiledb.stats_enable()
        tiledb.libtiledb.stats_reset()
        tiledb.libtiledb.stats_disable()

        tiledb.libtiledb.stats_enable()

        path = self.path("test_stats")

        with tiledb.from_numpy(path, np.arange(10)) as T:
            pass

        # basic output check for read stats
        tiledb.libtiledb.stats_reset()
        with tiledb.open(path) as T:
            tiledb.libtiledb.stats_enable()
            assert_array_equal(T, np.arange(10))

            # test stdout version
            tiledb.stats_dump()
            assert_captured(capfd, "TileDB Embedded Version:")

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

    def test_stats_include_python_json(self):
        tiledb.libtiledb.stats_enable()

        path = self.path("test_stats")

        with tiledb.from_numpy(path, np.arange(10)) as T:
            pass

        tiledb.libtiledb.stats_reset()
        with tiledb.open(path) as T:
            tiledb.libtiledb.stats_enable()
            assert_array_equal(T, np.arange(10))
            json_stats = tiledb.stats_dump(print_out=False, json=True)
            assert isinstance(json_stats, dict)
            assert "python" in json_stats
            assert "timers" in json_stats
            assert "counters" in json_stats
