from typing import TYPE_CHECKING

import tiledb.cc as lt

from .ctx import CtxMixin

if TYPE_CHECKING:
    from libtiledb import Ctx


class ArrayImpl(CtxMixin, lt.Array):
    """
    Represents a TileDB array.
    """

    def __init__(self, array, ctx: "Ctx" = None):
        """Class representing a TileDB Array.

        :param array: Cython TileDB Array class.
        :param ctx: A TileDB Context
        """
        super().__init__(ctx, array)
