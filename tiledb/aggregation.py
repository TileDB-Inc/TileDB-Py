import tiledb.libtiledb as lt


class Aggregation:
    """
    Proxy object returned by Query.agg to calculate aggregations.
    """

    def __init__(self, query=None, attr_to_aggs={}):
        if query is None:
            raise ValueError("must pass in a query object")

        self.query = query
        self.attr_to_aggs = attr_to_aggs

    def __getitem__(self, selection):
        from .main import PyAgg
        from .subarray import Subarray

        array = self.query.array
        order = self.query.order

        layout = (
            lt.LayoutType.UNORDERED if array.schema.sparse else lt.LayoutType.ROW_MAJOR
        )
        if order is None or order == "C":
            layout = lt.LayoutType.ROW_MAJOR
        elif order == "F":
            layout = lt.LayoutType.COL_MAJOR
        elif order == "G":
            layout = lt.LayoutType.GLOBAL_ORDER
        elif order == "U":
            layout = lt.LayoutType.UNORDERED
        else:
            raise ValueError(
                "order must be 'C' (TILEDB_ROW_MAJOR), "
                "'F' (TILEDB_COL_MAJOR), "
                "'G' (TILEDB_GLOBAL_ORDER), "
                "or 'U' (TILEDB_UNORDERED)"
            )

        q = PyAgg(array.ctx, array, layout, self.attr_to_aggs)

        from .array import (
            index_as_tuple,
            index_domain_subarray,
            replace_ellipsis,
            replace_scalars_slice,
        )

        selection = index_as_tuple(selection)
        dom = array.schema.domain
        idx = replace_ellipsis(dom.ndim, selection)
        idx, drop_axes = replace_scalars_slice(dom, idx)
        dim_ranges = index_domain_subarray(array, dom, idx)

        subarray = Subarray(array, array.ctx)
        subarray.add_ranges(dim_ranges)
        q.set_subarray(subarray)

        cond = self.query.cond
        if cond is not None and cond != "":
            from .query_condition import QueryCondition

            if isinstance(cond, str):
                q.set_cond(QueryCondition(cond))
            else:
                raise TypeError("`cond` expects type str.")

        result = q.get_aggregate()

        # If there was only one attribute, just show the aggregate results
        if len(result) == 1:
            result = result[list(result.keys())[0]]

            # If there was only one aggregate, just show the value
            if len(result) == 1:
                result = result[list(result.keys())[0]]

        return result

    @property
    def multi_index(self):
        """Apply Array.multi_index with query parameters."""
        from .multirange_indexing import MultiRangeAggregation

        return MultiRangeAggregation(self.query.array, query=self)

    @property
    def df(self):
        raise NotImplementedError(".df indexer not supported for Aggregations")
