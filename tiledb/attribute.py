import io

import tiledb.cc as lt


class Attr(lt.Attribute):
    """
    Represents a TileDB attribute.
    """

    def __init__(self, type: lt.ObjectType, uri: str):
        super().__init__(type, uri)

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return False
        if self.name != other.name or self.dtype != other.dtype:
            return False
        return True

    # def dump(self):
    #     """Dumps a string representation of the Attr object to standard output (stdout)"""
    #     check_error(self.ctx,
    #                 tiledb_attribute_dump(self.ctx.ptr, self.ptr, stdout))
    #     print('\n')
    #     return

    # @property
    # def dtype(self):
    #     """Return numpy dtype object representing the Attr type

    #     :rtype: numpy.dtype

    #     """
    #     cdef tiledb_datatype_t typ
    #     check_error(self.ctx,
    #                 tiledb_attribute_get_type(self.ctx.ptr, self.ptr, &typ))
    #     cdef uint32_t ncells = 0
    #     check_error(self.ctx,
    #                 tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))

    #     return np.dtype(_numpy_dtype(typ, ncells))

    # @property
    # def name(self):
    #     """Attribute string name, empty string if the attribute is anonymous

    #     :rtype: str
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     internal_name = self._get_name()
    #     # handle __attr names from arrays written with libtiledb < 2
    #     if internal_name == "__attr":
    #         return u""
    #     return internal_name

    # @property
    # def _internal_name(self):
    #     return self._get_name()

    # @property
    # def isanon(self):
    #     """True if attribute is an anonymous attribute

    #     :rtype: bool

    #     """
    #     cdef unicode name = self._get_name()
    #     return name == u"" or name.startswith(u"__attr")

    # @property
    # def compressor(self):
    #     """String label of the attributes compressor and compressor level

    #     :rtype: tuple(str, int)
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     # <todo> do we want to reimplement this on top of new API?
    #     pass

    # @property
    # def filters(self):
    #     """FilterList of the TileDB attribute

    #     :rtype: tiledb.FilterList
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef tiledb_filter_list_t* filter_list_ptr = NULL
    #     cdef int rc = TILEDB_OK
    #     check_error(self.ctx,
    #                 tiledb_attribute_get_filter_list(self.ctx.ptr, self.ptr, &filter_list_ptr))

    #     return FilterList(PyCapsule_New(filter_list_ptr, "fl", NULL),
    #         is_capsule=True, ctx=self.ctx)

    # @property
    # def fill(self):
    #     """Fill value for unset cells of this attribute

    #     :rtype: depends on dtype
    #     :raises: :py:exc:`tiledb.TileDBERror`
    #     """
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

    # @property
    # def isnullable(self):
    #     """True if the attribute is nullable

    #     :rtype: bool
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef uint8_t nullable = 0
    #     cdef int rc = TILEDB_OK
    #     check_error(
    #         self.ctx,
    #         tiledb_attribute_get_nullable(self.ctx.ptr, self.ptr, &nullable))

    #     return <bint>nullable

    # @property
    # def isvar(self):
    #     """True if the attribute is variable length

    #     :rtype: bool
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef unsigned int ncells = self._cell_val_num()
    #     return ncells == TILEDB_VAR_NUM

    # @property
    # def ncells(self):
    #     """The number of cells (scalar values) for a given attribute value

    #     :rtype: int
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef unsigned int ncells = self._cell_val_num()
    #     assert (ncells != 0)
    #     return int(ncells)

    # @property
    # def isascii(self):
    #     """True if the attribute is TileDB dtype TILEDB_STRING_ASCII

    #     :rtype: bool
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     return self._get_type() == TILEDB_STRING_ASCII

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
