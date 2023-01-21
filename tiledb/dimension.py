import io
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .filter import Filter, FilterList
from .util import (
    dtype_range,
    dtype_to_tiledb,
    numpy_dtype,
    tiledb_type_is_datetime,
    tiledb_type_is_integer,
)


def _tiledb_cast_tile_extent(tile_extent: Any, dtype: np.dtype) -> np.array:
    """Given a tile extent value, cast it to np.array of the given numpy dtype."""
    # Special handling for datetime domains
    if dtype.kind == "M":
        date_unit = np.datetime_data(dtype)[0]
        if isinstance(tile_extent, np.timedelta64):
            extent_value = int(tile_extent / np.timedelta64(1, date_unit))
            tile_size_array = np.array(np.int64(extent_value), dtype=np.int64)
        else:
            tile_size_array = np.array(tile_extent, dtype=np.int64)
    else:
        tile_size_array = np.array(tile_extent, dtype=dtype)

    if tile_size_array.size != 1:
        raise ValueError("tile extent must be a scalar")

    return tile_size_array


def _tiledb_cast_domain(
    domain, tiledb_dtype: lt.DataType
) -> Tuple[np.generic, np.generic]:
    np_dtype = numpy_dtype(tiledb_dtype)

    if tiledb_type_is_datetime(tiledb_dtype):
        date_unit = np.datetime_data(np_dtype)[0]
        return (
            np.datetime64(domain[0], date_unit),
            np.datetime64(domain[1], date_unit),
        )

    if tiledb_dtype in (
        lt.DataType.STRING_ASCII,
        lt.DataType.STRING_UTF8,
        lt.DataType.BLOB,
    ):
        return domain

    return (np_dtype(domain[0]), np_dtype(domain[1]))


class Dim(CtxMixin, lt.Dimension):
    """
    Represents a TileDB dimension.
    """

    def __init__(
        self,
        name: str = "__dim_0",
        domain: Tuple[Any, Any] = None,
        tile: Any = None,
        filters: Union[FilterList, Sequence[Filter]] = None,
        dtype: np.dtype = np.uint64,
        var: bool = None,
        ctx: Optional[Ctx] = None,
    ):
        """Class representing a dimension of a TileDB Array.

        :param name: Dimension name, empty if anonymous
        :param domain: TileDB domain
        :param tile: Tile extent
        :param filters: List of filters to apply
        :param dtype: Dimension value datatype
        :param var: Dimension is variable-length (automatic for byte/string types)
        :param ctx: A TileDB Context
        :raises ValueError: invalid domain or tile extent
        :raises TypeError: invalid domain, tile extent, or dtype type
        :raises tiledb.TileDBError:
        """
        if var is not None:
            if var and np.dtype(dtype) not in (np.str_, np.bytes_):
                raise TypeError("'var=True' specified for non-str/bytes dtype")

        if domain is not None and len(domain) != 2:
            raise ValueError("invalid domain extent, must be a pair")

        domain_array = None
        tile_size_array = None

        if (isinstance(dtype, str) and dtype == "ascii") or np.dtype(dtype).kind == "S":
            # Handle var-len dom type (currently only TILEDB_STRING_ASCII)
            # The dims's dom is implicitly formed as coordinates are written.
            dim_datatype = lt.DataType.STRING_ASCII
        else:
            if dtype is not None:
                dtype = np.dtype(dtype)
                dtype_min, dtype_max = dtype_range(dtype)

                if domain == (None, None):
                    # this means to use the full extent of the type
                    domain = (dtype_min, dtype_max)
                elif (
                    domain[0] < dtype_min
                    or domain[0] > dtype_max
                    or domain[1] < dtype_min
                    or domain[1] > dtype_max
                ):
                    raise TypeError(
                        "invalid domain extent, domain cannot be safely"
                        f" cast to dtype {dtype!r}"
                    )

            domain_array = np.asarray(domain, dtype=dtype)
            domain_dtype = domain_array.dtype
            dim_datatype = dtype_to_tiledb(domain_dtype)

            # check that the domain type is a valid dtype (integer / floating)
            if (
                not np.issubdtype(domain_dtype, np.integer)
                and not np.issubdtype(domain_dtype, np.floating)
                and not domain_dtype.kind == "M"
            ):
                raise TypeError(f"invalid Dim dtype {domain_dtype!r}")

            if tiledb_type_is_datetime(dim_datatype):
                domain_array = domain_array.astype(dtype=np.int64)

            # if the tile extent is specified, cast
            if tile is not None:
                tile_size_array = _tiledb_cast_tile_extent(tile, domain_dtype)
                if tile_size_array.size != 1:
                    raise ValueError("tile extent must be a scalar")

        super().__init__(ctx, name, dim_datatype, domain_array, tile_size_array)

        if filters is not None:
            if isinstance(filters, FilterList):
                self._filters = filters
            else:
                self._filters = FilterList(filters)

    def __repr__(self) -> str:
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for f in self.filters:
                filters_str += repr(f) + ", "
            filters_str += "])"

        # for consistency, print `var=True` for string-like types
        varlen = "" if self.dtype not in (np.str_, np.bytes_) else ", var=True"
        return f"Dim(name={self.name!r}, domain={self.domain!s}, tile={self.tile!r}, dtype='{self.dtype!s}'{varlen}{filters_str})"

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

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other) -> bool:
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

    def __array__(self, dtype=None, **kw) -> np.array:
        if not self._integer_domain():
            raise TypeError(
                "conversion to numpy ndarray only valid for integer dimension domains"
            )
        lb, ub = self.domain
        return np.arange(int(lb), int(ub) + 1, dtype=dtype if dtype else self.dtype)

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype representation of the dimension type.

        :rtype: numpy.dtype

        """
        return np.dtype(numpy_dtype(self._tiledb_dtype))

    @property
    def name(self) -> str:
        """The dimension label string.

        Anonymous dimensions return a default string representation based on the dimension index.

        :rtype: str

        """
        return self._name

    @property
    def isvar(self) -> bool:
        """True if the dimension is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._ncell == lt.TILEDB_VAR_NUM()

    @property
    def isanon(self) -> bool:
        """True if the dimension is anonymous

        :rtype: bool

        """
        return self.name == "" or self.name.startswith("__dim")

    @property
    def filters(self) -> FilterList:
        """FilterList of the TileDB dimension

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return FilterList.from_pybind11(self._ctx, self._filters)

    @property
    def shape(self) -> Tuple["np.generic", "np.generic"]:
        """The shape of the dimension given the dimension's domain.

        **Note**: The shape is only valid for integer and datetime dimension domains.

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain

        """
        if not tiledb_type_is_integer(
            self._tiledb_dtype
        ) and not tiledb_type_is_datetime(self._tiledb_dtype):
            raise TypeError(
                "shape only valid for integer and datetime dimension domains"
            )
        return ((self._domain[1] - self._domain[0] + 1),)

    @property
    def size(self) -> int:
        """The size of the dimension domain (number of cells along dimension).

        :rtype: int
        :raises TypeError: floating point (inexact) domain

        """
        if not tiledb_type_is_integer(self._tiledb_dtype):
            raise TypeError("size only valid for integer dimension domains")
        return int(self.shape[0])

    @property
    def tile(self) -> np.generic:
        """The tile extent of the dimension.

        :rtype: numpy scalar or np.timedelta64

        """
        np_dtype = numpy_dtype(self._tiledb_dtype)

        if tiledb_type_is_datetime(self._tiledb_dtype):
            date_unit = np.datetime_data(self.dtype)[0]
            return np.timedelta64(self._tile, date_unit)

        if self._tiledb_dtype in (
            lt.DataType.STRING_ASCII,
            lt.DataType.STRING_UTF8,
            lt.DataType.BLOB,
        ):
            return self._tile

        return np_dtype(self._tile)

    @property
    def domain(self) -> Tuple["np.generic", "np.generic"]:
        """The dimension (inclusive) domain.

        The dimension's domain is defined by a (lower bound, upper bound) tuple.

        :rtype: tuple(numpy scalar, numpy scalar)

        """
        return _tiledb_cast_domain(self._domain, self._tiledb_dtype)
