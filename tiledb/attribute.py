import io
import numpy as np
import warnings

from collections import deque

import tiledb.cc as lt
from .ctx import default_ctx
from .filter import FilterList


class Attr(lt.Attribute):
    """
    Represents a TileDB attribute.
    """

    def __init__(
        self,
        name="",
        dtype=np.float64,
        fill=None,
        var=None,
        nullable=False,
        filters=None,
        ctx=None,
    ):
        """Class representing a TileDB array attribute.

        :param tiledb.Ctx ctx: A TileDB Context
        :param str name: Attribute name, empty if anonymous
        :param dtype: Attribute value datatypes
        :type dtype: numpy.dtype object or type or string
        :param nullable: Attribute is nullable
        :type bool:
        :param fill: Fill value for unset cells.
        :param var: Attribute is variable-length (automatic for byte/string types)
        :type dtype: bool
        :param filters: List of filters to apply
        :type filters: FilterList
        :raises TypeError: invalid dtype
        :raises: :py:exc:`tiledb.TileDBError`

        """
        self._ctx = ctx or default_ctx()
        _cctx = lt.Context(self._ctx, False)

        if isinstance(dtype, str) and dtype == "ascii":
            var = True
            _ncell = lt.TILEDB_VAR_NUM()
            _dtype = np.dtype("S0")
        else:
            _dtype = np.dtype(dtype)
            _ncell = None

            if _dtype.kind == "U":
                if var is not None and var == False:
                    _ncell = _dtype.itemsize
                _ncell = lt.TILEDB_VAR_NUM()
            elif _dtype.kind == "S":
                if var is not None and var == False and 0 < _dtype.itemsize:
                    _ncell = _dtype.itemsize
                elif var is None:
                    if 0 < _dtype.itemsize:
                        var = False
                        _ncell = _dtype.itemsize
                    else:
                        var = True
                        _ncell = lt.TILEDB_VAR_NUM()
                else:
                    if var and 0 < _dtype.itemsize:
                        warnings.warn(
                            f"Attr given `var=True` but `dtype` `{dtype}` is fixed; "
                            "setting `dtype=S0`. Hint: set `var=True` with `dtype=S0`, "
                            f"or `var=False`with `dtype={dtype}`",
                            DeprecationWarning,
                        )
                        _dtype = np.dtype("S0")

                    if _dtype.itemsize == 0:
                        if var == False:
                            warnings.warn(
                                f"Attr given `var=False` but `dtype` `S0` is var-length; "
                                "setting `var=True` and `dtype=S0`. Hint: set `var=False` "
                                "with `dtype=S0`, or `var=False` with a fixed-width "
                                "string `dtype=S<n>` where is  n>1",
                                DeprecationWarning,
                            )
                    var = True
                    _ncell = lt.TILEDB_VAR_NUM()
            elif _dtype.kind == "V":
                # handles n fixed-size record dtypes
                if _dtype.shape != ():
                    raise TypeError("nested sub-array numpy dtypes are not supported")
                # check that types are the same
                deq = deque(_dtype.fields.values())
                typ0, _ = deq.popleft()
                nfields = 1
                for (typ, _) in deq:
                    nfields += 1
                    if typ != typ0:
                        raise TypeError(
                            "heterogenous record numpy dtypes are not supported"
                        )
                _dtype = typ0
                _ncell = len(dtype.fields.values())

        super().__init__(_cctx, name, _dtype)

        if _ncell:
            self._ncell = _ncell

        var = var or False

        if self._ncell == lt.TILEDB_VAR_NUM() and not var:
            raise TypeError("dtype is not compatible with var-length attribute")

        if filters is not None:
            self._filters = FilterList(filters)

        if fill is not None:
            self._fill = np.array([fill], dtype=_dtype)

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
    def dtype(self):
        """Return numpy dtype object representing the Attr type

        :rtype: numpy.dtype

        """
        return self._numpy_dtype

    @property
    def name(self):
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
    def isanon(self):
        """True if attribute is an anonymous attribute

        :rtype: bool

        """
        return self._name == "" or self._name.startswith("__attr")

    @property
    def filters(self):
        """FilterList of the TileDB attribute

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return FilterList(self._filters)

    @property
    def fill(self):
        """Fill value for unset cells of this attribute

        :rtype: depends on dtype
        :raises: :py:exc:`tiledb.TileDBERror`
        """
        value = self._fill
        if self._tiledb_dtype in (lt.DataType.STRING_UTF8, lt.DataType.STRING_ASCII):
            return value.tobytes()
        return value

    @property
    def isnullable(self):
        """True if the attribute is nullable

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._nullable

    @property
    def isvar(self):
        """True if the attribute is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._ncell == lt.TILEDB_VAR_NUM()

    @property
    def ncells(self):
        """The number of cells (scalar values) for a given attribute value

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        assert self._ncell != 0
        return int(self._ncell)

    @property
    def isascii(self):
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

        attr_dtype = "ascii" if self.isascii else self.dtype

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
