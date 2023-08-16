import itertools
import re
import textwrap
import warnings

import numpy as np

import tiledb

from .common import (
    DiskTestCase,
    fx_sparse_cell_order,  # noqa: F401
)


class ReprTest(DiskTestCase):
    def test_attr_repr(self):
        attr = tiledb.Attr(name="itsanattr", dtype=np.float64)
        self.assertTrue(
            re.match(
                r"Attr\(name=[u]?'itsanattr', dtype='float64', var=False, nullable=False, enum_label=None\)",
                repr(attr),
            )
        )

        g = dict()
        exec("from tiledb import Attr; from numpy import float64", g)
        self.assertEqual(eval(repr(attr), g), attr)

    def test_dim_repr(self):
        dtype_set = [bytes, np.bytes_]
        opts = {
            None: None,
            "var": True,
            "domain": (None, None),
            "filters": [tiledb.GzipFilter()],
        }

        dim_test_imports = textwrap.dedent(
            """
            from tiledb import Dim, FilterList, GzipFilter
            import numpy
            from numpy import float64
            """
        )

        for dtype in dtype_set:
            opt_choices = [
                itertools.combinations(opts.keys(), r=n)
                for n in range(1, len(opts) + 1)
            ]
            for opt_set in itertools.chain(*opt_choices):
                opt_kwarg = {k: opts[k] for k in opt_set if k}
                g = dict()
                exec(dim_test_imports, g)

                dim = tiledb.Dim(name="d1", dtype=dtype, **opt_kwarg)
                self.assertEqual(eval(repr(dim), g), dim)

        # test datetime
        g = dict()
        exec(dim_test_imports, g)
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
            tile=2,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(eval(repr(dim), g), dim)

    def test_arrayschema_repr(self, fx_sparse_cell_order):  # noqa: F811
        filters = tiledb.FilterList([tiledb.ZstdFilter(-1)])
        for sparse in [False, True]:
            cell_order = fx_sparse_cell_order if sparse else None
            domain = tiledb.Domain(
                tiledb.Dim(domain=(1, 8), tile=2), tiledb.Dim(domain=(1, 8), tile=2)
            )
            a1 = tiledb.Attr("val", dtype="f8", filters=filters)
            orig_schema = tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=sparse, cell_order=cell_order
            )

            schema_repr = repr(orig_schema)
            g = dict()
            setup = "from tiledb import *\n" "import numpy as np\n"

            exec(setup, g)
            new_schema = None
            try:
                new_schema = eval(schema_repr, g)
            except Exception:
                warn_str = (
                    """Exception during ReprTest schema eval"""
                    + """, schema string was:\n"""
                    + """'''"""
                    + """\n{}\n'''""".format(schema_repr)
                )
                warnings.warn(warn_str)
                raise

            self.assertEqual(new_schema, orig_schema)

    def test_arrayschema_repr_hilbert(self):
        domain = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
        a = tiledb.Attr("a", dtype="f8")
        schema = tiledb.ArraySchema(
            domain=domain, attrs=(a,), cell_order="hilbert", sparse=True
        )

        assert schema.cell_order == "hilbert"
        assert schema.tile_order is None
