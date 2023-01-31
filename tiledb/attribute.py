import io
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .filter import Filter, FilterList
from .util import array_type_ncells, numpy_dtype, tiledb_type_is_datetime


class Attr(CtxMixin, lt.Attribute):
    """
    Represents a TileDB attribute.
    """

    def __init__(
        self,
        name: str = "",
        dtype: np.dtype = np.float64,
        fill: Any = None,
        var: bool = None,
        nullable: bool = False,
        filters: Union[FilterList, Sequence[Filter]] = None,
        ctx: Optional[Ctx] = None,
    ):
        """Class representing a TileDB array attribute.

        :param name: Attribute name, empty if anonymous
        :param dtype: Attribute value datatype
        :param fill: Fill value for unset cells
        :param var: Attribute is variable-length (automatic for byte/string types)
        :param nullable: Attribute is nullable
        :param filters: List of filters to apply
        :param ctx: A TileDB Context
        :raises TypeError: invalid dtype
        :raises tiledb.TileDBError:
        """
        _dtype = None
        if isinstance(dtype, str) and dtype == "ascii":
            tiledb_dtype = lt.DataType.STRING_ASCII
            _ncell = lt.TILEDB_VAR_NUM()
            if var is None:
                var = True
        elif isinstance(dtype, str) and dtype == "blob":
            tiledb_dtype = lt.DataType.BLOB
            _ncell = 1
        else:
            _dtype = np.dtype(dtype)
            tiledb_dtype, _ncell = array_type_ncells(_dtype)

        # ensure that all unicode strings are var-length
        if var or (_dtype and _dtype.kind == "U"):
            var = True
            _ncell = lt.TILEDB_VAR_NUM()

        if _dtype and _dtype.kind == "S":
            if var and 0 < _dtype.itemsize:
                warnings.warn(
                    f"Attr given `var=True` but `dtype` `{_dtype}` is fixed; "
                    "setting `dtype=S0`. Hint: set `var=True` with `dtype=S0`, "
                    f"or `var=False`with `dtype={_dtype}`",
                    DeprecationWarning,
                )
                _dtype = np.dtype("S0")

            if _dtype.itemsize == 0:
                if var is False:
                    warnings.warn(
                        "Attr given `var=False` but `dtype` `S0` is var-length; "
                        "setting `var=True` and `dtype=S0`. Hint: set `var=False` "
                        "with `dtype=S0`, or `var=False` with a fixed-width "
                        "string `dtype=S<n>` where is  n>1",
                        DeprecationWarning,
                    )
                var = True
                _ncell = lt.TILEDB_VAR_NUM()

        var = var or False

        super().__init__(ctx, name, tiledb_dtype)

        if _ncell:
            self._ncell = _ncell

        var = var or False

        if self._ncell == lt.TILEDB_VAR_NUM() and not var:
            raise TypeError("dtype is not compatible with var-length attribute")

        if filters is not None:
            if isinstance(filters, FilterList):
                self._filters = filters
            else:
                self._filters = FilterList(filters)

        if fill is not None:
            if self._tiledb_dtype == lt.DataType.STRING_UTF8:
                self._fill = np.array([fill.encode("utf-8")], dtype="S")
            else:
                self._fill = np.array([fill], dtype=self.dtype)

        if nullable is not None:
            self._nullable = nullable

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return False
        if self.name != other.name or self.dtype != other.dtype:
            return False
        return True

    def dump(self):
        """Dumps a string representation of the Attr object to standard output (stdout)"""
        self._dump()

    @property
    def dtype(self) -> np.dtype:
        """Return numpy dtype object representing the Attr type

        :rtype: numpy.dtype

        """
        return np.dtype(numpy_dtype(self._tiledb_dtype, self._ncell))

    @property
    def name(self) -> str:
        """Attribute string name, empty string if the attribute is anonymous

        :rtype: str
        :raises: :py:exc:`tiledb.TileDBError`

        """
        internal_name = self._name
        # handle __attr names from arrays written with libtiledb < 2
        if internal_name == "__attr":
            return ""
        return internal_name

    @property
    def _internal_name(self):
        return self._name

    @property
    def isanon(self) -> bool:
        """True if attribute is an anonymous attribute

        :rtype: bool

        """
        return self._name == "" or self._name.startswith("__attr")

    @property
    def filters(self) -> FilterList:
        """FilterList of the TileDB attribute

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return FilterList.from_pybind11(self._ctx, self._filters)

    def _get_fill(self, value, dtype) -> Any:
        if dtype in (lt.DataType.CHAR, lt.DataType.BLOB):
            return value.tobytes()

        if dtype == lt.DataType.STRING_UTF8:
            return b"".join(value).decode("utf-8")

        if tiledb_type_is_datetime(dtype):
            return value[0].astype(np.timedelta64)

        return value

    @property
    def fill(self) -> Any:
        """Fill value for unset cells of this attribute

        :rtype: depends on dtype
        :raises: :py:exc:`tiledb.TileDBERror`
        """
        return self._get_fill(self._fill, self._tiledb_dtype)

    @property
    def isnullable(self) -> bool:
        """True if the attribute is nullable

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._nullable

    @property
    def isvar(self) -> bool:
        """True if the attribute is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._var

    @property
    def ncells(self) -> int:
        """The number of cells (scalar values) for a given attribute value

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        assert self._ncell != 0
        return int(self._ncell)

    @property
    def isascii(self) -> bool:
        """True if the attribute is TileDB dtype TILEDB_STRING_ASCII

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._tiledb_dtype == lt.DataType.STRING_ASCII

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for f in self.filters:
                filters_str += repr(f) + ", "
            filters_str += "])"

        if self._tiledb_dtype == lt.DataType.STRING_ASCII:
            attr_dtype = "ascii"
        elif self._tiledb_dtype == lt.DataType.BLOB:
            attr_dtype = "blob"
        else:
            attr_dtype = self.dtype

        # filters_str must be last with no spaces
        return (
            f"""Attr(name={repr(self.name)}, dtype='{attr_dtype!s}', """
            f"""var={self.isvar!s}, nullable={self.isnullable!s}"""
            f"""{filters_str})"""
        )

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>Is Nullable</th>")
        output.write("<th>Filters</th>")
        output.write("</tr>")
        output.write(f"{self._repr_html_row_only_()}")
        output.write("</table>")

        return output.getvalue()

    def _repr_html_row_only_(self):
        output = io.StringIO()

        output.write("<tr>")
        output.write(f"<td>{self.name}</td>")
        output.write(f"<td>{'ascii' if self.isascii else self.dtype}</td>")
        output.write(f"<td>{self.isvar}</td>")
        output.write(f"<td>{self.isnullable}</td>")
        output.write(f"<td>{self.filters._repr_html_()}</td>")
        output.write("</tr>")

        return output.getvalue()
