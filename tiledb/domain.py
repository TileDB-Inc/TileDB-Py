import tiledb.cc as lt


class Domain(lt.Domain):
    """
    Represents a TileDB domain.
    """

    def __init__(self, type: lt.ObjectType, uri: str):
        super().__init__(type, uri)

    def __repr__(self):
        dims = ",\n       ".join([repr(self.dim(i)) for i in range(self.ndim)])
        return "Domain({0!s})".format(dims)

    def _repr_html_(self) -> str:
        output = io.StringIO()

        output.write("<table>")

        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Domain</th>")
        output.write("<th>Tile</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-length</th>")
        output.write("<th>Filters</th>")
        output.write("</tr>")
        for i in range(self.ndim):
            output.write(self.dim(i)._repr_html_row_only_())
        output.write("</table>")

        return output.getvalue()

    def __len__(self):
        """Returns the number of dimensions of the domain"""
        return self.ndim

    def __iter__(self):
        """Returns a generator object that iterates over the domain's dimension objects"""
        return (self.dim(i) for i in range(self.ndim))

    def __eq__(self, other):
        """Returns true if Domain is equal to self.

        :rtype: bool
        """
        if not isinstance(other, Domain):
            return False

        same_dtype = self._is_homogeneous()

        if same_dtype and self.shape != other.shape:
            return False

        ndim = self.ndim
        if ndim != other.ndim:
            return False

        for i in range(ndim):
            if self.dim(i) != other.dim(i):
                return False
        return True

    # @property
    # def ndim(self):
    #     """The number of dimensions of the domain.

    #     :rtype: int

    #     """
    #     ndim = 0
    #     check_error(self.ctx,
    #                 tiledb_domain_get_ndim(self.ctx.ptr, self.ptr, &ndim))
    #     return ndim

    # @property
    # def dtype(self):
    #     """The numpy dtype of the domain's dimension type.

    #     :rtype: numpy.dtype

    #     """
    #     cdef tiledb_datatype_t typ = self._get_type()
    #     return np.dtype(_numpy_dtype(typ))

    # @property
    # def shape(self):
    #     """The domain's shape, valid only for integer domains.

    #     :rtype: tuple
    #     :raises TypeError: floating point (inexact) domain

    #     """
    #     if not self._integer_domain():
    #         raise TypeError("shape valid only for integer domains")
    #     return self._shape()

    # @property
    # def size(self):
    #     """The domain's size (number of cells), valid only for integer domains.

    #     :rtype: int
    #     :raises TypeError: floating point (inexact) domain

    #     """
    #     if not self._integer_domain():
    #         raise TypeError("shape valid only for integer domains")
    #     return np.product(self._shape())

    # @property
    # def homogeneous(self):
    #     """Returns True if the domain's dimension types are homogeneous."""
    #     return self._is_homogeneous()

    # def dim(self, dim_id):
    #     """Returns a Dim object from the domain given the dimension's index or name.

    #     :param dim_d: dimension index (int) or name (str)
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef tiledb_dimension_t* dim_ptr = NULL
    #     cdef bytes uname
    #     cdef const char* name_ptr = NULL

    #     if isinstance(dim_id, (str, unicode)):
    #         uname = ustring(dim_id).encode('UTF-8')
    #         name_ptr = uname
    #         check_error(self.ctx,
    #                     tiledb_domain_get_dimension_from_name(
    #                         self.ctx.ptr, self.ptr, name_ptr, &dim_ptr))
    #     elif isinstance(dim_id, int):
    #         check_error(self.ctx,
    #                     tiledb_domain_get_dimension_from_index(
    #                         self.ctx.ptr, self.ptr, dim_id, &dim_ptr))
    #     else:
    #         raise ValueError("Unsupported dim identifier: '{}' (expected int or str)".format(
    #             safe_repr(dim_id)
    #         ))

    #     assert(dim_ptr != NULL)
    #     return Dim.from_ptr(dim_ptr, self.ctx)

    # def has_dim(self, name):
    #     """
    #     Returns true if the Domain has a Dimension with the given name

    #     :param name: name of Dimension
    #     :rtype: bool
    #     :return:
    #     """
    #     cdef:
    #         cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
    #         cdef tiledb_domain_t* dom_ptr = self.ptr
    #         int32_t has_dim = 0
    #         int32_t rc = TILEDB_OK
    #         bytes bname = name.encode("UTF-8")

    #     rc = tiledb_domain_has_dimension(
    #         ctx_ptr,
    #         dom_ptr,
    #         bname,
    #         &has_dim
    #     )
    #     if rc != TILEDB_OK:
    #         _raise_ctx_err(ctx_ptr, rc)
    #     return bool(has_dim)

    # def dump(self):
    #     """Dumps a string representation of the domain object to standard output (STDOUT)"""
    #     check_error(self.ctx,
    #                 tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))
    #     print("\n")
    #     return
