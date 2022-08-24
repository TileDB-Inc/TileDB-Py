import io
import numpy as np

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

        super().__init__(_cctx, name, np.dtype(dtype))

        if filters is not None:
            self._filters = FilterList(filters)

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
        return self._fill_value

    #     cdef const uint8_t* value_ptr = NULL
    #     cdef uint64_t size
    #     check_error(self.ctx,
    #         tiledb_attribute_get_fill_value(
    #             self.ctx.ptr, self.ptr, <const void**>&value_ptr, &size))

    #     if value_ptr == NULL:
    #         return None

    #     if size == 0:
    #         raise TileDBError("Unexpected zero-length non-null fill value")

    #     cdef np.npy_intp shape[1]
    #     shape[0] = <np.npy_intp> 1
    #     cdef tiledb_datatype_t tiledb_type = self._get_type()
    #     cdef int typeid = _numpy_typeid(tiledb_type)
    #     assert(typeid != np.NPY_NOTYPE)
    #     cdef np.ndarray fill_array

    #     if np.issubdtype(self.dtype, np.bytes_):
    #         return (<char*>value_ptr)[:size]
    #     elif np.issubdtype(self.dtype, np.unicode_):
    #         return (<char*>value_ptr)[:size].decode('utf-8')
    #     else:
    #         fill_array = np.empty(1, dtype=self.dtype)
    #         memcpy(np.PyArray_DATA(fill_array), value_ptr, size)

    #     if _tiledb_type_is_datetime(tiledb_type):
    #         # Coerce to np.int64
    #         fill_array.dtype = np.int64
    #         datetime_dtype = _tiledb_type_to_datetime(tiledb_type).dtype
    #         date_unit = np.datetime_data(datetime_dtype)[0]
    #         tmp_val = None
    #         if fill_array[0] == 0:
    #             # undefined should span the whole dimension domain
    #             tmp_val = int(self.shape[0])
    #         else:
    #             tmp_val = int(fill_array[0])
    #         return np.timedelta64(tmp_val, date_unit)

    #     return fill_array

    @property
    def isnullable(self):
        """True if the attribute is nullable

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._var

    @property
    def isvar(self):
        """True if the attribute is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._ncell == lt.TILEDB_VAR_NUM

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
