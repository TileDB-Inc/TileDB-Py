import io
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .datatypes import DataType
from .filter import Filter, FilterList


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
        enum_label: str = None,
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
        dt = DataType.from_numpy(
            np.dtype(dtype) if dtype not in ("ascii", "blob") else dtype
        )

        # ensure that all strings are var-length
        if (var is None and dtype == "ascii") or np.issubdtype(dt.np_dtype, np.str_):
            var = True
        elif np.issubdtype(dt.np_dtype, np.bytes_):
            if dt.np_dtype.itemsize > 0 and var:
                warnings.warn(
                    f"Attr given `var=True` but `dtype` `{dtype}` is fixed; "
                    "setting `dtype=S0`. Hint: set `var=True` with `dtype=S0`, "
                    f"or `var=False`with `dtype={dtype}`",
                    DeprecationWarning,
                )
            elif dt.np_dtype.itemsize == 0 and dtype != "ascii":
                if var is False:
                    warnings.warn(
                        "Attr given `var=False` but `dtype` `S0` is var-length; "
                        "setting `var=True` and `dtype=S0`. Hint: set `var=False` "
                        "with `dtype=S0`, or `var=False` with a fixed-width "
                        "string `dtype=S<n>` where is  n>1",
                        DeprecationWarning,
                    )
                var = True

        super().__init__(ctx, name, dt.tiledb_type)

        if var:
            self._ncell = lt.TILEDB_VAR_NUM()
        elif dt.ncells != lt.TILEDB_VAR_NUM():
            self._ncell = dt.ncells
        else:
            raise TypeError("dtype is not compatible with var-length attribute")

        if filters is not None:
            if isinstance(filters, FilterList):
                self._filters = filters
            else:
                self._filters = FilterList(filters)

        if fill is not None:
            if self._tiledb_dtype == lt.DataType.STRING_UTF8:
                self._fill = np.array([fill.encode("utf-8")], dtype="S")
            elif self.dtype == np.dtype("complex64") or self.dtype == np.dtype(
                "complex128"
            ):
                if hasattr(fill, "dtype") and fill.dtype in {
                    np.dtype("f4, f4"),
                    np.dtype("f8, f8"),
                }:
                    _fill = fill["f0"] + fill["f1"] * 1j
                elif hasattr(fill, "__len__") and len(fill) == 2:
                    _fill = fill[0] + fill[1] * 1j
                else:
                    _fill = fill
                self._fill = np.array(_fill, dtype=self.dtype)
            else:
                self._fill = np.array([fill], dtype=self.dtype)

        if nullable is not None:
            self._nullable = nullable

        if enum_label is not None:
            self._set_enumeration_name(self._ctx, enum_label)

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return False
        if self.isnullable != other.isnullable or self.dtype != other.dtype:
            return False
        if not self.isnullable:
            # Check the fill values are equal.
            def equal_or_nan(x, y):
                return x == y or (np.isnan(x) and np.isnan(y))

            if self.ncells == 1:
                if not equal_or_nan(self.fill, other.fill):
                    return False
            elif np.issubdtype(self.dtype, np.bytes_) or np.issubdtype(
                self.dtype, np.str_
            ):
                if self.fill != other.fill:
                    return False
            elif self.dtype in {np.dtype("complex64"), np.dtype("complex128")}:
                if not (
                    equal_or_nan(np.real(self.fill), np.real(other.fill))
                    and equal_or_nan(np.imag(self.fill), np.imag(other.fill))
                ):
                    return False
            else:
                if not all(
                    equal_or_nan(x, y)
                    or (
                        isinstance(x, str)
                        and x.lower() == "nat"
                        and isinstance(y, str)
                        and y.lower() == "nat"
                    )
                    for x, y in zip(self.fill[0], other.fill[0])
                ):
                    return False
        return (
            self._internal_name == other._internal_name
            and self.isvar == other.isvar
            and self.filters == other.filters
        )

    def dump(self):
        """Dumps a string representation of the Attr object to standard output (stdout)"""
        self._dump()

    @property
    def dtype(self) -> np.dtype:
        """Return numpy dtype object representing the Attr type

        :rtype: numpy.dtype

        """
        return DataType.from_tiledb(self._tiledb_dtype, self._ncell).np_dtype

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

    @property
    def fill(self) -> Any:
        """Fill value for unset cells of this attribute

        :rtype: depends on dtype
        :raises: :py:exc:`tiledb.TileDBERror`
        """
        dtype = self.dtype
        if np.issubdtype(dtype, np.bytes_):
            return self._fill.tobytes()
        if np.issubdtype(dtype, np.str_):
            return self._fill.tobytes().decode("utf-8")
        if np.issubdtype(dtype, np.datetime64):
            return self._fill[0].astype(np.timedelta64)
        return self._fill

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

    @property
    def enum_label(self):
        return self._get_enumeration_name(self._ctx)

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

        if self.enum_label is None:
            enum_label = None
        else:
            enum_label = f"'{self.enum_label!s}'"

        # filters_str must be last with no spaces
        return (
            f"""Attr(name={repr(self.name)}, dtype='{attr_dtype!s}', """
            f"""var={self.isvar!s}, nullable={self.isnullable!s}, """
            f"""enum_label={enum_label}{filters_str})"""
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
