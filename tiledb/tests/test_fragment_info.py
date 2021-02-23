import tiledb
from tiledb.tests.common import DiskTestCase

import itertools
import numpy as np


class FragmentInfoTest(DiskTestCase):
    def setUp(self):
        super(FragmentInfoTest, self).setUp()
        if not tiledb.libtiledb.version() >= (2, 2):
            self.skipTest("Only run FragmentInfo test with TileDB>=2.2")

    def test_uri_dne(self):
        fragment_info = tiledb.fragment_info("does_not_exist")
        with self.assertRaises(tiledb.TileDBError):
            fragment_info.load()

    def test_dense_fragments(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_dense_fragments")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(0, 2), tile=fragments, dtype=np.int64), ctx=ctx
        )
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        fragment_info = tiledb.fragment_info(uri)

        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1
            with tiledb.DenseArray(uri, mode="w", timestamp=timestamp, ctx=ctx) as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

            fragment_info.load()

            self.assertEqual(fragment_info.fragment_num(), fragment_idx + 1)

        all_expected_uris = []
        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1

            self.assertEqual(
                fragment_info.timestamp_range(fragment_idx), (timestamp, timestamp)
            )

            if uri[0] != "/":
                uri = "/" + uri.replace("\\", "/")

            expected_uri = "file://{uri}/__{ts}_{ts}".format(uri=uri, ts=timestamp)
            actual_uri = fragment_info.fragment_uri(fragment_idx)

            all_expected_uris.append(expected_uri)

            self.assertTrue(actual_uri.startswith(expected_uri))
            self.assertTrue(
                actual_uri.endswith(str(fragment_info.version(fragment_idx)))
            )

            self.assertTrue(fragment_info.dense(fragment_idx))
            self.assertFalse(fragment_info.sparse(fragment_idx))

        all_actual_uris = fragment_info.fragment_uri()
        for actual_uri, expected_uri in zip(all_actual_uris, all_expected_uris):
            self.assertTrue(actual_uri.startswith(expected_uri))
            self.assertTrue(
                actual_uri.endswith(str(fragment_info.version(fragment_idx)))
            )

        self.assertEqual(fragment_info.timestamp_range(), ((1, 1), (2, 2), (3, 3)))
        self.assertEqual(fragment_info.dense(), (True, True, True))
        self.assertEqual(fragment_info.sparse(), (False, False, False))
        if tiledb.libtiledb.version() < (2, 2, 3):
            self.assertEqual(fragment_info.version(), (7, 7, 7))
        else:
            self.assertEqual(fragment_info.version(), (8, 8, 8))

    def test_sparse_fragments(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_sparse_fragments")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(0, 2), tile=fragments, dtype=np.int64), ctx=ctx
        )
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        fragment_info = tiledb.fragment_info(uri)

        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1
            with tiledb.SparseArray(uri, mode="w", timestamp=timestamp, ctx=ctx) as T:
                T[fragment_idx] = fragment_idx

            fragment_info.load()

            self.assertEqual(fragment_info.fragment_num(), fragment_idx + 1)

        all_expected_uris = []
        for fragment_idx in range(fragments):
            timestamp = fragment_idx + 1

            self.assertEqual(
                fragment_info.timestamp_range(fragment_idx), (timestamp, timestamp)
            )

            if uri[0] != "/":
                uri = "/" + uri.replace("\\", "/")

            expected_uri = "file://{uri}/__{ts}_{ts}".format(uri=uri, ts=timestamp)
            actual_uri = fragment_info.fragment_uri(fragment_idx)

            all_expected_uris.append(expected_uri)

            self.assertTrue(actual_uri.startswith(expected_uri))
            self.assertTrue(
                actual_uri.endswith(str(fragment_info.version(fragment_idx)))
            )

            self.assertFalse(fragment_info.dense(fragment_idx))
            self.assertTrue(fragment_info.sparse(fragment_idx))

        all_actual_uris = fragment_info.fragment_uri()
        for actual_uri, expected_uri in zip(all_actual_uris, all_expected_uris):
            self.assertTrue(actual_uri.startswith(expected_uri))
            self.assertTrue(
                actual_uri.endswith(str(fragment_info.version(fragment_idx)))
            )

        self.assertEqual(fragment_info.timestamp_range(), ((1, 1), (2, 2), (3, 3)))
        self.assertEqual(fragment_info.dense(), (False, False, False))
        self.assertEqual(fragment_info.sparse(), (True, True, True))
        if tiledb.libtiledb.version() < (2, 2, 3):
            self.assertEqual(fragment_info.version(), (7, 7, 7))
        else:
            self.assertEqual(fragment_info.version(), (8, 8, 8))

    def test_non_empty_domain(self):
        uri = self.path("test_non_empty_domain")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="x", ctx=ctx, domain=(1, 4)),
            tiledb.Dim(name="y", ctx=ctx, domain=(-2.0, 2.0), dtype=np.float32),
            ctx=ctx,
        )
        att = tiledb.Attr(ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            coords = np.array(
                list(itertools.product(np.arange(1, 5), np.arange(-1, 3)))
            )
            x = coords[:, 0]
            y = coords[:, 1]
            T[x, y] = np.array(range(16))

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            x = [1, 3]
            y = [-1.5, -1.25]
            T[x, y] = np.array(range(2))

        fragment_info = tiledb.fragment_info(uri, ctx)
        fragment_info.load()

        x_dt = schema.domain.dim(0).dtype
        y_dt = schema.domain.dim(1).dtype

        self.assertEqual(fragment_info.get_non_empty_domain(schema, 0, 0), (1, 4))
        self.assertEqual(fragment_info.get_non_empty_domain(schema, 0, 1), (-1.0, 2.0))
        self.assertEqual(fragment_info.get_non_empty_domain(schema, 1, 0), (1, 3))
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1, 1), (-1.5, -1.25)
        )

        self.assertEqual(fragment_info.get_non_empty_domain(schema, 0, "x"), (1, 4))
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 0, "y"), (-1.0, 2.0)
        )
        self.assertEqual(fragment_info.get_non_empty_domain(schema, 1, "x"), (1, 3))
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1, "y"), (-1.5, -1.25)
        )

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 0), ((1, 4), (-1.0, 2.0))
        )
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1), ((1, 3), (-1.5, -1.25))
        )

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema),
            (((1, 4), (-1.0, 2.0)), ((1, 3), (-1.5, -1.25))),
        )

    def test_non_empty_domain_date(self):
        uri = self.path("test_non_empty_domain")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(
            tiledb.Dim(
                name="day",
                ctx=ctx,
                domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
                dtype="datetime64[D]",
            ),
            ctx=ctx,
        )
        att = tiledb.Attr(ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            dates = np.array(
                ["2017-04-01", "2019-10-02", "2019-10-03", "2019-12-04"],
                dtype="datetime64[D]",
            )
            T[dates] = np.array(range(4))

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            dates = np.array(
                ["2010-01-01", "2013-10-02", "2014-10-03"], dtype="datetime64[D]"
            )
            T[dates] = np.array(range(3))

        fragment_info = tiledb.fragment_info(uri, ctx)
        fragment_info.load()

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 0, 0),
            (np.datetime64("2017-04-01"), np.datetime64("2019-12-04")),
        )
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1, 0),
            (np.datetime64("2010-01-01"), np.datetime64("2014-10-03")),
        )

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 0, "day"),
            (np.datetime64("2017-04-01"), np.datetime64("2019-12-04")),
        )
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1, "day"),
            (np.datetime64("2010-01-01"), np.datetime64("2014-10-03")),
        )

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 0),
            ((np.datetime64("2017-04-01"), np.datetime64("2019-12-04")),),
        )
        self.assertEqual(
            fragment_info.get_non_empty_domain(schema, 1),
            ((np.datetime64("2010-01-01"), np.datetime64("2014-10-03")),),
        )

        self.assertEqual(
            fragment_info.get_non_empty_domain(schema),
            (
                ((np.datetime64("2017-04-01"), np.datetime64("2019-12-04")),),
                ((np.datetime64("2010-01-01"), np.datetime64("2014-10-03")),),
            ),
        )

    def test_non_empty_domain_strings(self):
        uri = self.path("test_non_empty_domain_strings")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="x", ctx=ctx, domain=(None, None), dtype=np.bytes_),
            tiledb.Dim(name="y", ctx=ctx, domain=(None, None), dtype=np.bytes_),
            ctx=ctx,
        )
        att = tiledb.Attr(ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            x_dims = [b"a", b"b", b"c", b"d"]
            y_dims = [b"e", b"f", b"g", b"h"]
            T[x_dims, y_dims] = np.array([1, 2, 3, 4])

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            x_dims = [b"a", b"b"]
            y_dims = [b"e", b"f"]
            T[x_dims, y_dims] = np.array([1, 2])

        fragment_info = tiledb.fragment_info(uri, ctx)
        fragment_info.load()

        self.assertEqual(fragment_info.non_empty_domain_var(schema, 0, 0), ("a", "d"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 0, 1), ("e", "h"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 1, 0), ("a", "b"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 1, 1), ("e", "f"))

        self.assertEqual(fragment_info.non_empty_domain_var(schema, 0, "x"), ("a", "d"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 0, "y"), ("e", "h"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 1, "x"), ("a", "b"))
        self.assertEqual(fragment_info.non_empty_domain_var(schema, 1, "y"), ("e", "f"))

        self.assertEqual(
            fragment_info.non_empty_domain_var(schema, 0), (("a", "d"), ("e", "h"))
        )
        self.assertEqual(
            fragment_info.non_empty_domain_var(schema, 1), (("a", "b"), ("e", "f"))
        )

        self.assertEqual(
            fragment_info.non_empty_domain_var(schema),
            ((("a", "d"), ("e", "h")), (("a", "b"), ("e", "f"))),
        )

    def test_cell_num(self):
        uri = self.path("test_cell_num")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(1, 4)), ctx=ctx)
        att = tiledb.Attr(ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, sparse=True, domain=dom, attrs=(att,))

        tiledb.SparseArray.create(uri, schema)

        fragment_info = tiledb.fragment_info(uri, ctx)

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            a = np.array([1, 2, 3, 4])
            T[a] = a

        with tiledb.SparseArray(uri, mode="w", ctx=ctx) as T:
            b = np.array([1, 2])
            T[b] = b

        fragment_info = tiledb.fragment_info(uri, ctx)
        fragment_info.load()

        self.assertEqual(fragment_info.cell_num(0), len(a))
        self.assertEqual(fragment_info.cell_num(1), len(b))
        self.assertEqual(fragment_info.cell_num(), (len(a), len(b)))

    def test_consolidated_fragment_metadata(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_consolidated_fragment_metadata")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), dtype=np.int64), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        fragment_info = tiledb.fragment_info(uri, ctx)

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w", ctx=ctx) as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

        fragment_info.load()

        self.assertEqual(fragment_info.unconsolidated_metadata_num(), 3)
        for fragment_idx in range(fragments):
            self.assertFalse(fragment_info.has_consolidated_metadata(fragment_idx))

        self.assertEqual(
            fragment_info.has_consolidated_metadata(), (False, False, False)
        )

        tiledb.consolidate(
            uri,
            config=tiledb.Config(params={"sm.consolidation.mode": "fragment_meta"}),
            ctx=ctx,
        )

        fragment_info.load()

        self.assertEqual(fragment_info.unconsolidated_metadata_num(), 0)
        for fragment_idx in range(fragments):
            self.assertTrue(fragment_info.has_consolidated_metadata(fragment_idx))

        self.assertEqual(fragment_info.has_consolidated_metadata(), (True, True, True))

    def test_fragments_to_vacuum(self):
        fragments = 3

        A = np.zeros(fragments)

        uri = self.path("test_fragments_to_vacuum")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), dtype=np.int64), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))

        tiledb.DenseArray.create(uri, schema)

        fragment_info = tiledb.fragment_info(uri, ctx)

        for fragment_idx in range(fragments):
            with tiledb.DenseArray(uri, mode="w", ctx=ctx) as T:
                T[fragment_idx : fragment_idx + 1] = fragment_idx

        fragment_info.load()

        expected_vacuum_uri = fragment_info.fragment_uri(0)

        tiledb.consolidate(
            uri, config=tiledb.Config(params={"sm.vacuum.mode": "fragments"}), ctx=ctx
        )

        fragment_info.load()

        self.assertEqual(fragment_info.to_vacuum_num(), 3)
        self.assertEqual(fragment_info.to_vacuum_uri(0), expected_vacuum_uri)
        self.assertEqual(fragment_info.to_vacuum_uri(), (expected_vacuum_uri,))

        tiledb.vacuum(uri, ctx=ctx)

        fragment_info.load()

        self.assertEqual(fragment_info.to_vacuum_num(), 0)
