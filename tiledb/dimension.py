import io
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .datatypes import DataType
from .filter import Filter, FilterList


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
        dt = DataType.from_numpy(dtype)
        dtype = dt.np_dtype
        if dtype.kind not in ("i", "u", "f", "M", "S"):
            raise TypeError(f"invalid Dim dtype {dtype!r}")

        if var and not np.issubdtype(dtype, np.character):
            raise TypeError("'var=True' specified for non-str/bytes dtype")

        if domain is not None and len(domain) != 2:
            raise ValueError("invalid domain extent, must be a pair")

        if np.issubdtype(dtype, np.bytes_):
            # Handle var-len dom type (currently only TILEDB_STRING_ASCII)
            # The dims's dom is implicitly formed as coordinates are written.
            tiledb_type = lt.DataType.STRING_ASCII
            # XXX: intentionally(?) ignore passed domain and tile
            domain = tile = None
        else:
            tiledb_type = dt.tiledb_type
            if domain is None or domain == (None, None):
                domain = dt.domain
            else:
                dtype_min, dtype_max = dt.domain
                if not (
                    dtype_min <= domain[0] <= dtype_max
                    and dtype_min <= domain[1] <= dtype_max
                ):
                    raise TypeError(
                        f"invalid domain extent, domain cannot be safely cast to dtype {dtype!r}"
                    )

            domain = np.asarray(domain, dtype)
            if np.issubdtype(dtype, np.datetime64):
                domain = domain.astype(np.int64)

            if tile is not None:
                tile = dt.cast_tile_extent(tile)

        super().__init__(ctx, name, tiledb_type, domain, tile)

        if filters is not None:
            if isinstance(filters, FilterList):
                self._filters = filters
            else:
                self._filters = FilterList(filters)

    def __repr__(self) -> str:
        # use safe repr if pybind11 constructor failed
        if self._ctx is None:
            return object.__repr__(self)

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
        return (
            self.name == other.name
            and self.domain == other.domain
            and self.tile == other.tile
            and self.dtype == other.dtype
            and self.isvar == other.isvar
            and self.filters == other.filters
        )

    def __array__(self, dtype=None, **kw) -> np.array:
        if not self._integer_domain():
            raise TypeError(
                "conversion to numpy ndarray only valid for integer dimension domains"
            )
        lb, ub = self.domain
        return np.arange(int(lb), int(ub) + 1, dtype=dtype if dtype else self.dtype)

    def create_label_schema(
        self,
        order: str = "increasing",
        dtype: np.dtype = np.uint64,
        tile: Any = None,
        filters: Union[FilterList, Sequence[Filter]] = None,
    ):
        """Creates a dimension label schema for a dimension label on this dimension

        :param order: Order or sort of the label data ('increasing' or 'decreasing').
        :param dtype: Datatype of the label data.
        :param tile: Tile extent for the dimension of the dimension label. If
            ``None``, it will use the tile extent of this dimension.
        :param label_filters: Filter list for the attribute storing the label data.

        :rtype: DimLabelSchema

        """
        from .dimension_label_schema import DimLabelSchema

        return DimLabelSchema(
            order,
            dtype,
            self.dtype,
            self.tile if tile is None and self.tile != 0 else tile,
            filters,
            self._ctx,
        )

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype representation of the dimension type.

        :rtype: numpy.dtype

        """
        return DataType.from_tiledb(self._tiledb_dtype).np_dtype

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
        dtype = self.dtype
        if not (
            np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.datetime64)
        ):
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
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("size only valid for integer dimension domains")
        return int(self.shape[0])

    @property
    def tile(self) -> np.generic:
        """The tile extent of the dimension.

        :rtype: numpy scalar or np.timedelta64

        """
        dim_dtype = DataType.from_tiledb(self._tiledb_dtype)
        return dim_dtype.uncast_tile_extent(self._tile)

    @property
    def domain(self) -> Tuple["np.generic", "np.generic"]:
        """The dimension (inclusive) domain.

        The dimension's domain is defined by a (lower bound, upper bound) tuple.

        :rtype: tuple(numpy scalar, numpy scalar)

        """
        np_dtype = self.dtype
        if np.issubdtype(np_dtype, np.character):
            return self._domain

        min_args = [self._domain[0]]
        max_args = [self._domain[1]]
        if np.issubdtype(np_dtype, np.datetime64):
            unit = np.datetime_data(np_dtype)[0]
            min_args.append(unit)
            max_args.append(unit)
        return np_dtype.type(*min_args), np_dtype.type(*max_args)
