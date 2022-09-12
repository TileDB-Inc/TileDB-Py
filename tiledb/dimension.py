import io
from typing import TYPE_CHECKING
import numpy as np

import tiledb.cc as lt
from .ctx import default_ctx
from .filter import FilterList
from .util import _tiledb_type_is_integer, _tiledb_type_is_datetime

if TYPE_CHECKING:
    from .libtiledb import Ctx


class Dim(lt.Dimension):
    """
    Represents a TileDB dimension.
    """

    def __init__(
        self,
        name: str = "__dim_0",
        domain=None,
        tile=None,
        filters=None,
        dtype: np.dtype = np.uint64,
        var: bool = None,
        ctx: "Ctx" = None,
        _lt_obj=None,
    ):
        """Class representing a dimension of a TileDB Array.

        :param str name: the dimension name, empty if anonymous
        :param domain:
        :type domain: tuple(int, int) or tuple(float, float)
        :param tile: Tile extent
        :type tile: int or float
        :param filters: List of filters to apply
        :type filters: FilterList
        :dtype: the Dim numpy dtype object, type object, or string \
            that can be corerced into a numpy dtype object
        :raises ValueError: invalid domain or tile extent
        :raises TypeError: invalid domain, tile extent, or dtype type
        :raises: :py:exc:`TileDBError`
        :param tiledb.Ctx ctx: A TileDB Context

        """
        self._ctx = ctx or default_ctx()
        _cctx = lt.Context(self._ctx, False)

        if _lt_obj is not None:
            name = _lt_obj._name
            dtype = _lt_obj._numpy_dtype
            domain = np.array(_lt_obj._domain)
            tile = _lt_obj._tile
            filters = _lt_obj._filters

        if (isinstance(dtype, str) and dtype == "ascii") or np.dtype(dtype).kind == "S":
            # Handle var-len dom type (currently only TILEDB_STRING_ASCII)
            # The dims's dom is implicitly formed as coordinates are written.
            dtype = np.dtype("S")

        domain = np.array(domain)

        if np.issubdtype(np.dtype(dtype), np.datetime64) and not np.issubdtype(
            domain.dtype, np.datetime64
        ):
            raise TypeError("datetime dimension must have datetime domain")

        if np.issubdtype(domain.dtype, np.datetime64):
            domain = np.array(domain, dtype=np.uint64)
            tile = np.array(tile, dtype=np.uint64)
        else:
            domain = np.array(domain, dtype=dtype)
            tile = np.array(tile, dtype=dtype)

        super().__init__(_cctx, name, np.dtype(dtype), domain, tile)

        if filters is not None:
            self._filters = FilterList(filters)

    # def __repr__(self):
    #     filters_str = ""
    #     if self.filters:
    #         filters_str = ", filters=FilterList(["
    #         for f in self.filters:
    #             filters_str += repr(f) + ", "
    #         filters_str += "])"

    #     # for consistency, print `var=True` for string-like types
    #     varlen = "" if not self.dtype in (np.str_, np.bytes_) else ", var=True"
    #     return "Dim(name={0!r}, domain={1!s}, tile={2!r}, dtype='{3!s}'{4}{5})".format(
    #         self.name, self.domain, self.tile, self.dtype, varlen, filters_str
    #     )

    def _repr_html_(self) -> str:
        output = io.StringIO()

        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Domain</th>")
        output.write("<th>Tile</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>Filters</th>")
        output.write("</tr>")
        output.write(self._repr_html_row_only_())
        output.write("</table>")

        return output.getvalue()

    def _repr_html_row_only_(self) -> str:
        output = io.StringIO()

        output.write("<tr>")
        output.write(f"<td>{self.name}</td>")
        output.write(f"<td>{self.domain}</td>")
        output.write(f"<td>{self.tile}</td>")
        output.write(f"<td>{self.dtype}</td>")
        output.write(f"<td>{self.dtype in (np.str_, np.bytes_)}</td>")
        output.write(f"<td>{self.filters._repr_html_()}</td>")
        output.write("</tr>")

        return output.getvalue()

    def __len__(self):
        return self.size

    def __eq__(self, other):
        if not isinstance(other, Dim):
            return False
        if (
            self.name != other.name
            or self.domain != other.domain
            or self.tile != other.tile
            or self.dtype != other.dtype
        ):
            return False
        return True

    def __array__(self, dtype=None, **kw):
        if not self._integer_domain():
            raise TypeError(
                "conversion to numpy ndarray only valid for integer dimension domains"
            )
        lb, ub = self.domain
        return np.arange(int(lb), int(ub) + 1, dtype=dtype if dtype else self.dtype)

    @property
    def dtype(self):
        """Numpy dtype representation of the dimension type.

        :rtype: numpy.dtype

        """
        return self._numpy_dtype

    @property
    def name(self):
        """The dimension label string.

        Anonymous dimensions return a default string representation based on the dimension index.

        :rtype: str

        """
        return self._name

    @property
    def isvar(self):
        """True if the dimension is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._ncell == lt.TILEDB_VAR_NUM()

    @property
    def isanon(self):
        """True if the dimension is anonymous

        :rtype: bool

        """
        return self.name == "" or self.name.startswith("__dim")

    @property
    def filters(self):
        """FilterList of the TileDB dimension

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return FilterList(self._filters)

    @property
    def shape(self):
        """The shape of the dimension given the dimension's domain.

        **Note**: The shape is only valid for integer and datetime dimension domains.

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain

        """
        if not _tiledb_type_is_integer(
            self._tiledb_dtype
        ) and not _tiledb_type_is_datetime(self._tiledb_dtype):
            raise TypeError(
                "shape only valid for integer and datetime dimension domains"
            )
        return ((self._domain[1] - self._domain[0] + 1),)

    @property
    def size(self):
        """The size of the dimension domain (number of cells along dimension).

        :rtype: int
        :raises TypeError: floating point (inexact) domain

        """
        if not _tiledb_type_is_integer(self._tiledb_dtype):
            raise TypeError("size only valid for integer dimension domains")
        return int(self._ncell)

    @property
    def tile(self):
        """The tile extent of the dimension.

        :rtype: numpy scalar or np.timedelta64

        """
        if _tiledb_type_is_datetime(self._tiledb_dtype):
            date_unit = np.datetime_data(self._numpy_dtype)[0]
            return np.timedelta64(self._tile, date_unit)
        return self._tile

    @property
    def domain(self):
        """The dimension (inclusive) domain.

        The dimension's domain is defined by a (lower bound, upper bound) tuple.

        :rtype: tuple(numpy scalar, numpy scalar)

        """
        if _tiledb_type_is_datetime(self._tiledb_dtype):
            date_unit = np.datetime_data(self._numpy_dtype)[0]
            return (
                np.datetime64(self._domain[0], date_unit),
                np.datetime64(self._domain[1], date_unit),
            )
        return self._domain
