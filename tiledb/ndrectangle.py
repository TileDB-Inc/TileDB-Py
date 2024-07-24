from typing import Tuple, Union

import tiledb
import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .domain import Domain


class NDRectangle(CtxMixin, lt.NDRectangle):
    """
    Represents a TileDB N-Dimensional Rectangle.
    """

    _dim_types = {}

    def __init__(self, ctx: Ctx, domain: Domain):
        """Class representing an N-Dimensional Rectangle of a TileDB Domain.

        :param ctx: A TileDB Context
        :param domain: A TileDB Domain
        :raises tiledb.TileDBError:
        """
        super().__init__(ctx, domain)

    def set_range(
        self,
        dim: Union[str, int],
        start: Union[int, float, str],
        end: Union[int, float, str],
    ):
        """Sets a range for the given dimension.

        :param dim: Dimension name or index
        :param start: Range start value
        :param end: Range end value
        :raises tiledb.TileDBError:
        """
        # Set types only if range is set successfully
        try:
            self._set_range(dim, start, end)
            self._dim_types[dim] = type(start).__name__
        except Exception as e:
            raise e

    def range(
        self, dim: Union[str, int]
    ) -> Union[Tuple[int, int], Tuple[float, float], Tuple[str, str]]:
        """Gets the range for the given dimension.

        :param dim: Dimension name or index
        :return: Range as a tuple (start, end)
        :raises tiledb.TileDBError:
        """
        if dim not in self._dim_types:
            if isinstance(dim, int):
                raise tiledb.TileDBError(
                    "Trying to get a range for an index out of bounds is not possible"
                )
            elif isinstance(dim, str):
                raise tiledb.TileDBError(
                    "Cannot get dimension index; Invalid dimension name"
                )
        return tuple(self._range(dim, self._dim_types[dim]))
