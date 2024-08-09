import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import tiledb

from .common import DiskTestCase

all_filter_types = [
    tiledb.NoOpFilter,
    tiledb.GzipFilter,
    tiledb.ZstdFilter,
    tiledb.LZ4Filter,
    tiledb.RleFilter,
    tiledb.Bzip2Filter,
    tiledb.DeltaFilter,
    tiledb.DoubleDeltaFilter,
    tiledb.DictionaryFilter,
    tiledb.BitWidthReductionFilter,
    tiledb.BitShuffleFilter,
    tiledb.ByteShuffleFilter,
    tiledb.PositiveDeltaFilter,
    tiledb.ChecksumSHA256Filter,
    tiledb.ChecksumMD5Filter,
    tiledb.FloatScaleFilter,
]


def filter_applicable(filter_type, attr_type) -> bool:
    """Return bool indicating filter applicability to a given attribute type."""
    if not isinstance(attr_type, type):
        # guard issubclass below: first argument must be a type
        return True
    elif issubclass(attr_type, np.floating) and filter_type in [
        tiledb.DoubleDeltaFilter
    ]:
        return False

    return True


class TestFilterTest(DiskTestCase):
    def test_filter(self):
        gzip_filter = tiledb.GzipFilter(level=10)
        self.assertIsInstance(gzip_filter, tiledb.Filter)
        self.assertEqual(gzip_filter.level, 10)

        bw_filter = tiledb.BitWidthReductionFilter(window=10)
        self.assertIsInstance(bw_filter, tiledb.Filter)
        self.assertEqual(bw_filter.window, 10)

        filter_list = tiledb.FilterList([gzip_filter, bw_filter], chunksize=1024)
        self.assertEqual(filter_list.chunksize, 1024)
        self.assertEqual(len(filter_list), 2)
        self.assertEqual(filter_list[0].level, gzip_filter.level)
        self.assertEqual(filter_list[1].window, bw_filter.window)

        # test filter list iteration
        self.assertEqual(len(list(filter_list)), 2)

        # test `filters` kwarg accepts python list of filters
        tiledb.Attr("foo", dtype=np.int64, filters=[gzip_filter])
        tiledb.Attr("foo", dtype=np.int64, filters=(gzip_filter,))

        attr = tiledb.Attr("foo", dtype=np.int64, filters=filter_list)

        self.assertEqual(len(attr.filters), 2)
        self.assertEqual(attr.filters.chunksize, filter_list.chunksize)

    @pytest.mark.parametrize("attr_type", [np.int64])
    @pytest.mark.parametrize("filter_type", all_filter_types)
    def test_filter_list(self, attr_type, filter_type):
        if not filter_applicable(filter_type, attr_type):
            pytest.mark.skip("Filter not supported for attribute type '{attr_type}'")

        # should be constructible without a `filters` keyword arg set
        filter_list1 = tiledb.FilterList()
        filter_list1.append(filter_type())
        self.assertEqual(len(filter_list1), 1)
        repr(filter_list1)

        filter_list2 = [x for x in filter_list1]
        attr = tiledb.Attr(filters=filter_list2, dtype=attr_type)
        self.assertEqual(len(attr.filters), 1)

    @pytest.mark.parametrize(
        "filter_type,name",
        [
            (tiledb.NoOpFilter, "NONE"),
            (tiledb.GzipFilter, "GZIP"),
            (tiledb.ZstdFilter, "ZSTD"),
            (tiledb.LZ4Filter, "LZ4"),
            (tiledb.RleFilter, "RLE"),
            (tiledb.Bzip2Filter, "BZIP2"),
            (tiledb.DeltaFilter, "DELTA"),
            (tiledb.DoubleDeltaFilter, "DOUBLE_DELTA"),
            (tiledb.DictionaryFilter, "DICTIONARY"),
            (tiledb.BitWidthReductionFilter, "BIT_WIDTH_REDUCTION"),
            (tiledb.BitShuffleFilter, "BITSHUFFLE"),
            (tiledb.ByteShuffleFilter, "BYTESHUFFLE"),
            (tiledb.PositiveDeltaFilter, "POSITIVE_DELTA"),
            (tiledb.ChecksumSHA256Filter, "CHECKSUM_SHA256"),
            (tiledb.ChecksumMD5Filter, "CHECKSUM_MD5"),
            (tiledb.FloatScaleFilter, "SCALE_FLOAT"),
        ],
    )
    def test_filter_name(self, filter_type, name):
        assert filter_type().filter_name == name

    @pytest.mark.parametrize("filter", all_filter_types)
    def test_all_filters(self, filter):
        # test initialization

        # make sure that repr works and round-trips correctly
        # some of these have attributes, so we just check the class name here
        self.assertTrue(filter.__name__ in repr(filter))

        tmp_globals = dict()
        setup = "from tiledb import *"
        exec(setup, tmp_globals)

        filter_repr = repr(filter())
        new_filter = None
        try:
            new_filter = eval(filter_repr, tmp_globals)
        except Exception:
            warn_str = (
                """Exception during FilterTest filter repr eval"""
                + """, filter repr string was:\n"""
                + """'''"""
                + """\n{}\n'''""".format(filter_repr)
            )
            warnings.warn(warn_str)
            raise

        self.assertEqual(new_filter, filter())

    def test_dictionary_encoding(self):
        path = self.path("test_dictionary_encoding")
        dom = tiledb.Domain(tiledb.Dim(name="row", domain=(0, 9), dtype=np.uint64))
        attr = tiledb.Attr(
            dtype="ascii",
            var=True,
            filters=tiledb.FilterList([tiledb.DictionaryFilter()]),
        )
        schema = tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=True)
        tiledb.Array.create(path, schema)

        data = [b"x" * i for i in np.random.randint(1, 10, size=10)]

        with tiledb.open(path, "w") as A:
            A[np.arange(10)] = data

        with tiledb.open(path, "r") as A:
            assert_array_equal(A[:][""], data)

    @pytest.mark.parametrize("factor", [1, 0.5, 2])
    @pytest.mark.parametrize("offset", [0])
    @pytest.mark.parametrize("bytewidth", [1, 8])
    def test_float_scaling_filter(self, factor, offset, bytewidth):
        path = self.path("test_float_scaling_filter")
        dom = tiledb.Domain(tiledb.Dim(name="row", domain=(0, 9), dtype=np.uint64))

        filter = tiledb.FloatScaleFilter(factor, offset, bytewidth)

        attr = tiledb.Attr(dtype=np.float64, filters=tiledb.FilterList([filter]))
        schema = tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=True)
        tiledb.Array.create(path, schema)

        data = np.random.rand(10)

        with tiledb.open(path, "w") as A:
            A[np.arange(10)] = data

        with tiledb.open(path, "r") as A:
            filter = A.schema.attr("").filters[0]
            assert filter.factor == factor
            assert filter.offset == offset
            assert filter.bytewidth == bytewidth

            # TODO compute the correct tolerance here
            assert_allclose(data, A[:][""], rtol=1, atol=1)

    @pytest.mark.parametrize(
        "attr_dtype,reinterp_dtype,expected_reinterp_dtype",
        [
            (np.uint64, None, None),
            (np.float64, np.uint64, np.uint64),
            (np.float64, tiledb.cc.DataType.UINT64, np.uint64),
        ],
    )
    def test_delta_filter(self, attr_dtype, reinterp_dtype, expected_reinterp_dtype):
        path = self.path("test_delta_filter")

        dom = tiledb.Domain(tiledb.Dim(name="row", domain=(0, 9), dtype=np.uint64))

        if reinterp_dtype is None:
            filter = tiledb.DeltaFilter()
        else:
            filter = tiledb.DeltaFilter(reinterp_dtype=reinterp_dtype)
        assert filter.reinterp_dtype == expected_reinterp_dtype

        attr = tiledb.Attr(dtype=attr_dtype, filters=tiledb.FilterList([filter]))

        assert attr.filters[0].reinterp_dtype == expected_reinterp_dtype

        schema = tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=False)
        tiledb.Array.create(path, schema)

        data = np.random.randint(0, 10_000_000, size=10)
        if attr_dtype == np.float64:
            data = data.astype(np.float64)

        with tiledb.open(path, "w") as A:
            A[:] = data

        with tiledb.open(path) as A:
            res = A[:]
            assert_array_equal(res, data)

    @pytest.mark.parametrize(
        "attr_dtype,reinterp_dtype,expected_reinterp_dtype",
        [
            (np.uint64, None, None),
            (np.float64, np.uint64, np.uint64),
            (np.float64, tiledb.cc.DataType.UINT64, np.uint64),
        ],
    )
    def test_double_delta_filter(
        self, attr_dtype, reinterp_dtype, expected_reinterp_dtype
    ):
        path = self.path("test_delta_filter")

        dom = tiledb.Domain(tiledb.Dim(name="row", domain=(0, 9), dtype=np.uint64))

        if reinterp_dtype is None:
            filter = tiledb.DoubleDeltaFilter()
        else:
            filter = tiledb.DoubleDeltaFilter(reinterp_dtype=reinterp_dtype)
        assert filter.reinterp_dtype == expected_reinterp_dtype

        attr = tiledb.Attr(dtype=attr_dtype, filters=tiledb.FilterList([filter]))
        assert attr.filters[0].reinterp_dtype == expected_reinterp_dtype
        schema = tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=False)
        tiledb.Array.create(path, schema)

        data = np.random.randint(0, 10_000_000, size=10)
        if attr_dtype == np.float64:
            data = data.astype(np.float64)

        with tiledb.open(path, "w") as A:
            A[:] = data

        with tiledb.open(path) as A:
            res = A[:]
            assert_array_equal(res, data)
