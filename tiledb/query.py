import tiledb.cc as lt

from .ctx import Ctx, CtxMixin, default_ctx
from .libtiledb import Array
from .subarray import Subarray


class Query(CtxMixin, lt.Query):
    """
    Represents a TileDB query.
    """

    def __init__(
        self,
        array: Array,
        ctx: Ctx = None,
    ):
        """Class representing a query on a TileDB Array.

        :param array: tiledb.Array the query is on
        :param ctx: A TileDB Context
        """
        self._array = array
        super().__init__(
            ctx, lt.Array(ctx if ctx is not None else default_ctx(), array)
        )

    def subarray(self) -> Subarray:
        """Subarray with the ranges this query is on.

        :rtype: Subarray
        """
        return Subarray.from_pybind11(self._ctx, self._subarray)
