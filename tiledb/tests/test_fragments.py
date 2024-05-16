import itertools
import sys
import xml.etree.ElementTree

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb
from tiledb.main import PyFragmentInfo

from .common import DiskTestCase


class FragmentInfoTest(DiskTestCase):
    def setUp(self):
        super().setUp()
        if not tiledb.libtiledb.version() >= (2, 2):
            pytest.skip("Only run FragmentInfo test with TileDB>=2.2")

    def test_uri_dne(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.array_fragments("does_not_exist")

    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_array_fragments(self, use_timestamps):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_dense_fragments")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        if use_timestamps:
            for fragment_idx in range(fragments):
                timestamp = fragment_idx + 1
                with tiledb.DenseArray(uri, mode="w", timestamp=timestamp) as T:
                    T[fragment_idx : fragment_idx + 1] = fragment_idx
        else:
            for fragment_idx in range(fragments):
                with tiledb.DenseArray(uri, mode="w") as T:
                    T[fragment_idx : fragment_idx + 1] = fragment_idx

        fi = tiledb.array_fragments(uri)

        assert len(fi) == 3
        assert fi.unconsolidated_metadata_num == 3
        assert fi.cell_num == (3, 3, 3)
        assert fi.has_consolidated_metadata == (False, False, False)
        assert fi.nonempty_domain == (((0, 0),), ((1, 1),), ((2, 2),))
        assert fi.sparse == (False, False, False)
        if use_timestamps:  # timestamps cannot be predicted if not used on write
            assert fi.timestamp_range == ((1, 1), (2, 2), (3, 3))
        assert fi.to_vacuum == ()
        assert hasattr(fi, "version")  # don't pin to a specific version

        for idx, frag in enumerate(fi):
            assert frag.cell_num == 3
            assert frag.has_consolidated_metadata is False
            assert frag.nonempty_domain == ((idx, idx),)
            assert frag.sparse is False
            if use_timestamps:  # timestamps cannot be predicted if not used on write
                assert frag.timestamp_range == (idx + 1, idx + 1)
            assert hasattr(frag, "version")  # don't pin to a specific version
            try:
                assert xml.etree.ElementTree.fromstring(frag._repr_html_()) is not None
            except:
                pytest.fail(
                    f"Could not parse frag._repr_html_(). Saw {frag._repr_html_()}"
                )

        try:
            assert xml.etree.ElementTree.fromstring(fi._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse fi._repr_html_(). Saw {fi._repr_html_()}")

    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_array_fragments_var(self, use_timestamps):
        fragments = 3

        uri = self.path("test_array_fragments_var")
        dom = tiledb.Domain(
            tiledb.Dim(name="dim", domain=(None, None), tile=None, dtype=np.bytes_)
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="1s", dtype=np.int32, var=True)],
        )

        tiledb.SparseArray.create(uri, schema)

        for fragment_idx in range(fragments):

            data = np.array(
                [
                    np.array(
                        [fragment_idx + 1] * 1,
                        dtype=np.int32,
                    ),
                    np.array(
                        [fragment_idx + 1] * 2,
                        dtype=np.int32,
                    ),
                    np.array(
                        [fragment_idx + 1] * 3,
                        dtype=np.int32,
                    ),
                ],
                dtype="O",
            )

            with tiledb.SparseArray(
                uri, mode="w", timestamp=fragment_idx + 1 if use_timestamps else None
            ) as T:
                T[["zero", "one", "two"]] = data

        fragments_info = tiledb.array_fragments(uri)

        self.assertEqual(
            fragments_info.nonempty_domain,
            ((("one", "zero"),), (("one", "zero"),), (("one", "zero"),)),
        )

        for frag in fragments_info:
            self.assertEqual(frag.nonempty_domain, (("one", "zero"),))

    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_dense_fragments(self, use_timestamps):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_dense_fragments")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1 if use_timestamps else None
            with tiledb.DenseArray(uri, mode="w", timestamp=timestamp) as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

            fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
            self.assertEqual(fragment_info.get_num_fragments(), fragment_idx + 1)

        if use_timestamps:  # asserts are not predictable without timestamps
            all_expected_uris = []
            for fragment_idx in range(fragments):
                timestamp = fragment_idx + 1

                self.assertEqual(
                    fragment_info.get_timestamp_range()[fragment_idx],
                    (timestamp, timestamp),
                )

                expected_uri = f"__{timestamp}_{timestamp}"
                actual_uri = fragment_info.get_uri()[fragment_idx]

                all_expected_uris.append(expected_uri)

                self.assertTrue(expected_uri in actual_uri)
                self.assertTrue(
                    actual_uri.endswith(str(fragment_info.get_version()[fragment_idx]))
                )
                self.assertFalse(fragment_info.get_sparse()[fragment_idx])

            all_actual_uris = fragment_info.get_uri()
            for actual_uri, expected_uri in zip(all_actual_uris, all_expected_uris):
                self.assertTrue(expected_uri in actual_uri)
                self.assertTrue(
                    actual_uri.endswith(str(fragment_info.get_version()[fragment_idx]))
                )

            self.assertEqual(
                fragment_info.get_timestamp_range(), ((1, 1), (2, 2), (3, 3))
            )
            self.assertEqual(fragment_info.get_sparse(), (False, False, False))

    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_sparse_fragments(self, use_timestamps):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_sparse_fragments")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1 if use_timestamps else None
            with tiledb.SparseArray(uri, mode="w", timestamp=timestamp) as T:
                T[fragment_idx] = fragment_idx

            fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())
            self.assertEqual(fragment_info.get_num_fragments(), fragment_idx + 1)

        if use_timestamps:  # asserts are not predictable without timestamps
            all_expected_uris = []
            for fragment_idx in range(fragments):
                timestamp = fragment_idx + 1

                self.assertEqual(
                    fragment_info.get_timestamp_range()[fragment_idx],
                    (timestamp, timestamp),
                )

                if uri[0] != "/":
                    uri = "/" + uri.replace("\\", "/")

                expected_uri = f"/__{timestamp}_{timestamp}"
                actual_uri = fragment_info.get_uri()[fragment_idx]

                all_expected_uris.append(expected_uri)

                self.assertTrue(expected_uri in actual_uri)
                self.assertTrue(
                    actual_uri.endswith(str(fragment_info.get_version()[fragment_idx]))
                )
                self.assertTrue(fragment_info.get_sparse()[fragment_idx])

            all_actual_uris = fragment_info.get_uri()
            for actual_uri, expected_uri in zip(all_actual_uris, all_expected_uris):
                self.assertTrue(expected_uri in actual_uri)
                self.assertTrue(
                    actual_uri.endswith(str(fragment_info.get_version()[fragment_idx]))
                )

            self.assertEqual(
                fragment_info.get_timestamp_range(), ((1, 1), (2, 2), (3, 3))
            )
            self.assertEqual(fragment_info.get_sparse(), (True, True, True))

    def test_nonempty_domain(self):
        uri = self.path("test_nonempty_domain")
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(1, 4)),
            tiledb.Dim(name="y", domain=(-2.0, 2.0), dtype=np.float32),
        )
        att = tiledb.Attr()
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w") as T:
            coords = np.array(
                list(itertools.product(np.arange(1, 5), np.arange(-1, 3)))
            )
            x = coords[:, 0]
            y = coords[:, 1]
            T[x, y] = np.array(range(16))

        with tiledb.SparseArray(uri, mode="w") as T:
            x = [1, 3]
            y = [-1.5, -1.25]
            T[x, y] = np.array(range(2))

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(
            fragment_info.get_nonempty_domain(),
            (((1, 4), (-1.0, 2.0)), ((1, 3), (-1.5, -1.25))),
        )

    def test_nonempty_domain_date(self):
        uri = self.path("test_nonempty_domain")
        dom = tiledb.Domain(
            tiledb.Dim(
                name="day",
                domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
                dtype="datetime64[D]",
            )
        )
        att = tiledb.Attr()
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w") as T:
            dates = np.array(
                ["2017-04-01", "2019-10-02", "2019-10-03", "2019-12-04"],
                dtype="datetime64[D]",
            )
            T[dates] = np.array(range(4))

        with tiledb.SparseArray(uri, mode="w") as T:
            dates = np.array(
                ["2010-01-01", "2013-10-02", "2014-10-03"], dtype="datetime64[D]"
            )
            T[dates] = np.array(range(3))

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(
            fragment_info.get_nonempty_domain(),
            (
                ((np.datetime64("2017-04-01"), np.datetime64("2019-12-04")),),
                ((np.datetime64("2010-01-01"), np.datetime64("2014-10-03")),),
            ),
        )

    def test_nonempty_domain_strings(self):
        uri = self.path("test_nonempty_domain_strings")
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(None, None), dtype=np.bytes_),
            tiledb.Dim(name="y", domain=(None, None), dtype=np.bytes_),
        )
        att = tiledb.Attr()
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w") as T:
            x_dims = [b"a", b"b", b"c", b"d"]
            y_dims = [b"e", b"f", b"g", b"h"]
            T[x_dims, y_dims] = np.array([1, 2, 3, 4])

        with tiledb.SparseArray(uri, mode="w") as T:
            x_dims = [b"a", b"b"]
            y_dims = [b"e", b"f"]
            T[x_dims, y_dims] = np.array([1, 2])

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(
            fragment_info.get_nonempty_domain(),
            ((("a", "d"), ("e", "h")), (("a", "b"), ("e", "f"))),
        )

    def test_cell_num(self):
        uri = self.path("test_cell_num")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 4)))
        att = tiledb.Attr()
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        with tiledb.SparseArray(uri, mode="w") as T:
            a = np.array([1, 2, 3, 4])
            T[a] = a

        with tiledb.SparseArray(uri, mode="w") as T:
            b = np.array([1, 2])
            T[b] = b

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(fragment_info.get_cell_num(), (len(a), len(b)))

    def test_consolidated_fragment_metadata(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_consolidated_fragment_metadata")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w") as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(fragment_info.get_unconsolidated_metadata_num(), 3)
        self.assertEqual(
            fragment_info.get_has_consolidated_metadata(), (False, False, False)
        )

        tiledb.consolidate(
            uri, config=tiledb.Config(params={"sm.consolidation.mode": "fragment_meta"})
        )

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        self.assertEqual(fragment_info.get_unconsolidated_metadata_num(), 0)
        self.assertEqual(
            fragment_info.get_has_consolidated_metadata(), (True, True, True)
        )

    def test_fragments_to_vacuum(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_fragments_to_vacuum")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w") as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        expected_vacuum_uri = fragment_info.get_uri()[0]

        tiledb.consolidate(
            uri, config=tiledb.Config(params={"sm.vacuum.mode": "fragments"})
        )

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        assert len(fragment_info.get_to_vacuum()) == 3
        assert fragment_info.get_to_vacuum()[0] == expected_vacuum_uri

        tiledb.vacuum(uri)

        fragment_info = PyFragmentInfo(uri, schema, False, tiledb.default_ctx())

        assert len(fragment_info.get_to_vacuum()) == 0

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 5, 0),
        reason=(
            "MBRs in FragmentInfo only available in "
            "tiledb.libtiledb.version() < (2, 5, 0)"
        ),
    )
    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_get_mbr(self, use_timestamps):
        fragments = 3

        uri = self.path("test_get_mbr")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=fragments, dtype=np.int64))
        att = tiledb.Attr(dtype=np.uint64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.Array.create(uri, schema)

        for fragi in range(fragments):
            timestamp = fragi + 1
            with tiledb.open(
                uri, mode="w", timestamp=timestamp if use_timestamps else None
            ) as T:
                T[np.array(range(0, fragi + 1))] = [fragi] * (fragi + 1)

        expected_mbrs = ((((0, 0),),), (((0, 1),),), (((0, 2),),))

        py_fragment_info = PyFragmentInfo(uri, schema, True, tiledb.default_ctx())
        assert py_fragment_info.get_mbrs() == expected_mbrs

        array_fragments = tiledb.array_fragments(uri)
        with pytest.raises(AttributeError) as excinfo:
            array_fragments.mbrs
        assert "retrieving minimum bounding rectangles is disabled" in str(
            excinfo.value
        )

        with self.assertRaises(AttributeError):
            array_fragments[0].mbrs
        assert "retrieving minimum bounding rectangles is disabled" in str(
            excinfo.value
        )

        array_fragments = tiledb.array_fragments(uri, include_mbrs=True)
        assert array_fragments.mbrs == expected_mbrs
        assert array_fragments[0].mbrs == expected_mbrs[0]
        assert array_fragments[1].mbrs == expected_mbrs[1]
        assert array_fragments[2].mbrs == expected_mbrs[2]

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 5, 0),
        reason=(
            "MBRs in FragmentInfo only available in "
            "tiledb.libtiledb.version() < (2, 5, 0)"
        ),
    )
    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_get_var_sized_dim_mbrs(self, use_timestamps):
        fragments = 3

        uri = self.path("test_get_var_sized_dim_mbrs")
        dom = tiledb.Domain(tiledb.Dim(dtype="ascii"))
        att = tiledb.Attr(dtype=np.uint64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.Array.create(uri, schema)

        for fragi in range(fragments):
            timestamp = fragi + 1
            with tiledb.open(
                uri, mode="w", timestamp=timestamp if use_timestamps else None
            ) as T:
                coords = [chr(i) * (fragi + 1) for i in range(97, fragi + 98)]
                T[np.array(coords)] = [fragi] * (fragi + 1)

        expected_mbrs = (((("a", "a"),),), ((("aa", "bb"),),), ((("aaa", "ccc"),),))

        py_fragment_info = PyFragmentInfo(uri, schema, True, tiledb.default_ctx())
        assert py_fragment_info.get_mbrs() == expected_mbrs

        array_fragments = tiledb.array_fragments(uri)
        with pytest.raises(AttributeError) as excinfo:
            array_fragments.mbrs
        assert "retrieving minimum bounding rectangles is disabled" in str(
            excinfo.value
        )

        with self.assertRaises(AttributeError):
            array_fragments[0].mbrs
        assert "retrieving minimum bounding rectangles is disabled" in str(
            excinfo.value
        )

        array_fragments = tiledb.array_fragments(uri, include_mbrs=True)
        assert array_fragments.mbrs == expected_mbrs
        assert array_fragments[0].mbrs == expected_mbrs[0]
        assert array_fragments[1].mbrs == expected_mbrs[1]
        assert array_fragments[2].mbrs == expected_mbrs[2]


class CreateArrayFromFragmentsTest(DiskTestCase):
    @pytest.mark.skipif(
        sys.platform == "win32", reason="VFS.copy() does not run on windows"
    )
    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_create_array_from_fragments(self, use_timestamps):
        dshape = (1, 3)
        num_frags = 10

        def create_array(target_path, dshape):
            dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=len(dshape)))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path, dshape, num_frags):
            for i in range(1, num_frags + 1):
                with tiledb.open(
                    target_path, "w", timestamp=i if use_timestamps else None
                ) as A:
                    A[[1, 2, 3]] = np.random.rand(dshape[1])

        src_path = self.path("test_create_array_from_fragments_src")
        dst_path = self.path("test_create_array_from_fragments_dst")

        ts = tuple((t, t) for t in range(1, 11))

        create_array(src_path, dshape)
        write_fragments(src_path, dshape, num_frags)
        frags = tiledb.FragmentInfoList(src_path)
        assert len(frags) == 10
        if use_timestamps:
            assert frags.timestamp_range == ts

        if use_timestamps:
            tiledb.create_array_from_fragments(src_path, dst_path, (3, 6))
        else:
            tiledb.create_array_from_fragments(
                src_path,
                dst_path,
                (frags.timestamp_range[2][0], frags.timestamp_range[5][1]),
            )

        frags = tiledb.FragmentInfoList(dst_path)
        assert len(frags) == 4
        if use_timestamps:
            assert frags.timestamp_range == ts[2:6]


class CopyFragmentsToExistingArrayTest(DiskTestCase):
    @pytest.mark.skipif(
        sys.platform == "win32", reason="VFS.copy() does not run on windows"
    )
    def test_copy_fragments_to_existing_array(self):
        def create_array(target_path, dshape):
            dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=len(dshape)))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path, dshape, num_frags, ts_start=1):
            for i in range(ts_start, ts_start + num_frags):
                with tiledb.open(target_path, "w", timestamp=i) as A:
                    A[[1, 2, 3]] = np.random.rand(dshape[1])

        tiledb.VFS()

        src_dshape = (1, 3)
        src_num_frags = 10
        src_path = self.path("test_copy_fragments_to_existing_array_src")
        create_array(src_path, src_dshape)
        write_fragments(src_path, src_dshape, src_num_frags)

        dst_dshape = (1, 3)
        dst_num_frags = 10
        dst_path = self.path("test_copy_fragments_to_existing_array_dst")
        create_array(dst_path, dst_dshape)
        write_fragments(dst_path, dst_dshape, dst_num_frags, 11)

        ts = tuple((t, t) for t in range(1, 21))

        frags = tiledb.array_fragments(dst_path)
        assert len(frags) == 10
        assert frags.timestamp_range == ts[10:]

        tiledb.copy_fragments_to_existing_array(src_path, dst_path, (3, 6))

        frags = tiledb.FragmentInfoList(dst_path)
        assert len(frags) == 14
        assert frags.timestamp_range == ts[2:6] + ts[10:]

    @pytest.mark.skipif(
        sys.platform == "win32", reason="VFS.copy() does not run on windows"
    )
    def test_copy_fragments_to_existing_array_mismatch(self):
        def create_array(target_path, attr_type):
            dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=3))
            att = tiledb.Attr(dtype=attr_type)
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path):
            for i in range(10):
                with tiledb.open(target_path, "w") as A:
                    A[[1, 2, 3]] = np.random.rand(3)

        src_path = self.path("test_copy_fragments_to_existing_array_evolved_src")
        create_array(src_path, "int64")
        write_fragments(src_path)

        dst_path = self.path("test_copy_fragments_to_existing_array_evolved_dst")
        create_array(dst_path, "int32")
        write_fragments(dst_path)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.copy_fragments_to_existing_array(src_path, dst_path, (3, 6))

    @pytest.mark.skipif(
        sys.platform == "win32", reason="VFS.copy() does not run on windows"
    )
    def test_copy_fragments_to_existing_array_evolved(self):
        def create_array(target_path):
            dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=3))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path):
            for i in range(10):
                with tiledb.open(target_path, "w") as A:
                    A[[1, 2, 3]] = np.random.rand(3)

        src_path = self.path("test_copy_fragments_to_existing_array_evolved_src")
        create_array(src_path)
        write_fragments(src_path)

        dst_path = self.path("test_copy_fragments_to_existing_array_evolved_dst")
        create_array(dst_path)
        write_fragments(dst_path)

        ctx = tiledb.default_ctx()
        se = tiledb.ArraySchemaEvolution(ctx)
        se.add_attribute(tiledb.Attr("a2", dtype=np.float64))
        se.array_evolve(src_path)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.copy_fragments_to_existing_array(src_path, dst_path, (3, 6))


class DeleteFragmentsTest(DiskTestCase):
    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_delete_fragments(self, use_timestamps):
        dshape = (1, 3)
        num_writes = 10

        def create_array(target_path, dshape):
            dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=len(dshape)))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path, dshape, num_writes):
            for i in range(1, num_writes + 1):
                with tiledb.open(
                    target_path, "w", timestamp=i if use_timestamps else None
                ) as A:
                    A[[1, 2, 3]] = np.random.rand(dshape[1])

        path = self.path("test_delete_fragments")

        ts = tuple((t, t) for t in range(1, 11))

        create_array(path, dshape)
        write_fragments(path, dshape, num_writes)
        frags = tiledb.array_fragments(path)
        assert len(frags) == 10
        if use_timestamps:
            assert frags.timestamp_range == ts

        if use_timestamps:
            tiledb.Array.delete_fragments(path, 3, 6)
        else:
            tiledb.Array.delete_fragments(
                path, frags.timestamp_range[2][0], frags.timestamp_range[5][1]
            )

        frags = tiledb.array_fragments(path)
        assert len(frags) == 6
        if use_timestamps:
            assert frags.timestamp_range == ts[:2] + ts[6:]

    @pytest.mark.parametrize("use_timestamps", [True, False])
    def test_delete_fragments_with_schema_evolution(self, use_timestamps):
        path = self.path("test_delete_fragments_with_schema_evolution")
        dshape = (1, 3)

        dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=len(dshape)))
        att = tiledb.Attr(name="a1", dtype=np.float64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.libtiledb.Array.create(path, schema)

        ts1_data = np.random.rand(3)
        if use_timestamps:
            with tiledb.open(path, "w", timestamp=1) as A:
                A[[1, 2, 3]] = ts1_data
        else:
            with tiledb.open(path, "w") as A:
                A[[1, 2, 3]] = ts1_data

        ctx = tiledb.default_ctx()
        se = tiledb.ArraySchemaEvolution(ctx)
        se.add_attribute(tiledb.Attr("a2", dtype=np.float64))
        se.array_evolve(path)

        ts2_data = np.random.rand(3)
        if use_timestamps:
            with tiledb.open(path, "w", timestamp=2) as A:
                A[[1, 2, 3]] = {"a1": ts2_data, "a2": ts2_data}
        else:
            with tiledb.open(path, "w") as A:
                A[[1, 2, 3]] = {"a1": ts2_data, "a2": ts2_data}

        frags = tiledb.array_fragments(path)
        assert len(frags) == 2

        with tiledb.open(path, "r") as A:
            assert_array_equal(A[:]["a1"], ts2_data)
            assert_array_equal(A[:]["a2"], ts2_data)

        if use_timestamps:
            tiledb.Array.delete_fragments(path, 2, 2)
        else:
            tiledb.Array.delete_fragments(
                path, frags.timestamp_range[1][0], frags.timestamp_range[1][1]
            )

        frags = tiledb.array_fragments(path)
        assert len(frags) == 1

        with tiledb.open(path, "r") as A:
            assert_array_equal(A[:]["a1"], ts1_data)
            assert_array_equal(A[:]["a2"], [np.nan, np.nan, np.nan])
