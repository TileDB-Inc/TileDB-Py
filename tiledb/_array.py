from typing import TYPE_CHECKING

import tiledb.cc as lt

from .ctx import CtxMixin


if TYPE_CHECKING:
    from libtiledb import Ctx


class ArrayImpl(CtxMixin, lt.Array):
    """
    TODO: Documentation for PyArray
    """

    def __init__(self, array, ctx: "Ctx" = None):
        super().__init__(ctx, array)
