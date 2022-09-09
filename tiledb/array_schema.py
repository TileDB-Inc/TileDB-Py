from typing import TYPE_CHECKING

import tiledb.cc as lt
from .attribute import Attr
from .ctx import default_ctx
from .domain import Domain

import io
import numbers

if TYPE_CHECKING:
    from .libtiledb import Ctx


class ArraySchema(lt.ArraySchema):
    """
    Schema class for TileDB dense / sparse array representations

    :param domain: Domain of schema
    :type attrs: tuple(tiledb.Attr, ...)
    :param cell_order:  TileDB label for cell layout
    :type cell_order: 'row-major' (default) or 'C', 'col-major' or 'F' or 'hilbert'
    :param tile_order:  TileDB label for tile layout
    :type tile_order: 'row-major' (default) or 'C', 'col-major' or 'F'
    :param int capacity: tile cell capacity
    :param offsets_filters: (default None) offsets filter list
    :type offsets_filters: tiledb.FilterList
    :param validity_filters: (default None) validity filter list
    :type validity_filters: tiledb.FilterList
    :param bool allows_duplicates: True if duplicates are allowed
    :param bool sparse: True if schema is sparse, else False \
        (set by SparseArray and DenseArray derived classes)
    :param tiledb.Ctx ctx: A TileDB Context
    :raises: :py:exc:`tiledb.TileDBError`

    """

    def __init__(
        self,
        domain=None,
        attrs=(),
        cell_order="row-major",
        tile_order="row-major",
        capacity=0,
        coords_filters=None,
        offsets_filters=None,
        validity_filters=None,
        allows_duplicates=False,
        sparse=False,
        ctx: "Ctx" = None,
        _uri=None,
    ):
        _ctx = ctx or default_ctx()
        _cctx = lt.Context(_ctx, False)

        if _uri:
            super().__init__(_cctx, _uri)
        else:
            _type = lt.ArrayType.SPARSE if sparse else lt.ArrayType.DENSE

            super().__init__(_cctx, _type)

            if domain:
                self._domain = domain

            if attrs:
                for att in attrs:
                    self._add_attr(att)

    # @staticmethod
    # def load(uri, Ctx ctx=None, key=None):
    #     if not ctx:
    #         ctx = default_ctx()
    #     cdef bytes buri = uri.encode('UTF-8')
    #     cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    #     cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    #     cdef tiledb_array_schema_t* array_schema_ptr = NULL
    #     # encryption key
    #     cdef bytes bkey
    #     cdef tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
    #     cdef void* key_ptr = NULL
    #     cdef unsigned int key_len = 0
    #     if key is not None:
    #         if isinstance(key, str):
    #             bkey = key.encode('ascii')
    #         else:
    #             bkey = bytes(key)
    #         key_type = TILEDB_AES_256_GCM
    #         key_ptr = <void *> PyBytes_AS_STRING(bkey)
    #         #TODO: unsafe cast here ssize_t -> uint64_t
    #         key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
    #     cdef int rc = TILEDB_OK
    #     with nogil:
    #         rc = tiledb_array_schema_load_with_key(
    #             ctx_ptr, uri_ptr, key_type, key_ptr, key_len, &array_schema_ptr)
    #     if rc != TILEDB_OK:
    #         _raise_ctx_err(ctx_ptr, rc)
    #     return ArraySchema.from_ptr(array_schema_ptr, ctx=ctx)

    # def __eq__(self, other):
    #     """Instance is equal to another ArraySchema"""
    #     if not isinstance(other, ArraySchema):
    #         return False
    #     nattr = self.nattr
    #     if nattr != other.nattr:
    #         return False
    #     if (self.sparse != other.sparse or
    #         self.cell_order != other.cell_order or
    #         self.tile_order != other.tile_order):
    #         return False
    #     if (self.capacity != other.capacity):
    #         return False
    #     if self.domain != other.domain:
    #         return False
    #     if self.coords_filters != other.coords_filters:
    #         return False
    #     for i in range(nattr):
    #         if self.attr(i) != other.attr(i):
    #             return False
    #     return True

    # def __len__(self):
    #     """Returns the number of Attributes in the ArraySchema"""
    #     return self.nattr

    # def __iter__(self):
    #     """Returns a generator object that iterates over the ArraySchema's Attribute objects"""
    #     return (self.attr(i) for i in range(self.nattr))

    # def check(self):
    #     """Checks the correctness of the array schema

    #     :rtype: None
    #     :raises: :py:exc:`tiledb.TileDBError` if invalid
    #     """
    #     check_error(self.ctx,
    #                 tiledb_array_schema_check(self.ctx.ptr, self.ptr))

    @property
    def sparse(self):
        """True if the array is a sparse array representation

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._array_type == lt.ArrayType.SPARSE

    # @property
    # def allows_duplicates(self):
    #     """Returns True if the (sparse) array allows duplicates."""

    #     if not self.sparse:
    #         raise TileDBError("ArraySchema.allows_duplicates does not apply to dense arrays")

    #     cdef int ballows_dups
    #     tiledb_array_schema_get_allows_dups(self.ctx.ptr, self.ptr, &ballows_dups)
    #     return bool(ballows_dups)

    # @property
    # def capacity(self):
    #     """The array capacity

    #     :rtype: int
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef uint64_t cap = 0
    #     check_error(self.ctx,
    #                 tiledb_array_schema_get_capacity(self.ctx.ptr, self.ptr, &cap))
    #     return cap

    # @property
    # def cell_order(self):
    #     """The cell order layout of the array."""
    #     cdef tiledb_layout_t order = TILEDB_UNORDERED
    #     self._cell_order(&order)
    #     return _tiledb_layout_string(order)

    # cdef _tile_order(ArraySchema self, tiledb_layout_t* tile_order_ptr):
    #     check_error(self.ctx,
    #         tiledb_array_schema_get_tile_order(self.ctx.ptr, self.ptr, tile_order_ptr))

    # @property
    # def tile_order(self):
    #     """The tile order layout of the array.

    #     :rtype: str
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     cdef tiledb_layout_t order = TILEDB_UNORDERED
    #     self._tile_order(&order)

    #     layout_string = _tiledb_layout_string(order)
    #     if self.cell_order == "hilbert":
    #         layout_string = None

    #     return layout_string

    # @property
    # def coords_compressor(self):
    #     """The compressor label and level for the array's coordinates.

    #     :rtype: tuple(str, int)
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     # <todo> reimplement on top of filter API?
    #     pass

    # @property
    # def offsets_compressor(self):
    #     """The compressor label and level for the array's variable-length attribute offsets.

    #     :rtype: tuple(str, int)
    #     :raises: :py:exc:`tiledb.TileDBError`

    #     """
    #     # <todo> reimplement on top of filter API?
    #     pass

    # @property
    # def offsets_filters(self):
    #     """The FilterList for the array's variable-length attribute offsets

    #     :rtype: tiledb.FilterList
    #     :raises: :py:exc:`tiledb.TileDBError`
    #     """
    #     cdef tiledb_filter_list_t* filter_list_ptr = NULL
    #     check_error(self.ctx,
    #         tiledb_array_schema_get_offsets_filter_list(
    #             self.ctx.ptr, self.ptr, &filter_list_ptr))
    #     return FilterList(
    #         PyCapsule_New(filter_list_ptr, "fl", NULL),
    #             is_capsule=True, ctx=self.ctx)

    # @property
    # def coords_filters(self):
    #     """The FilterList for the array's coordinates

    #     :rtype: tiledb.FilterList
    #     :raises: :py:exc:`tiledb.TileDBError`
    #     """
    #     cdef tiledb_filter_list_t* filter_list_ptr = NULL
    #     check_error(self.ctx,
    #         tiledb_array_schema_get_coords_filter_list(
    #             self.ctx.ptr, self.ptr, &filter_list_ptr))
    #     return FilterList(
    #         PyCapsule_New(filter_list_ptr, "fl", NULL),
    #             is_capsule=True, ctx=self.ctx)

    # @coords_filters.setter
    # def coords_filters(self, value):
    #     warnings.warn(
    #         "coords_filters is deprecated; "
    #         "set the FilterList for each dimension",
    #         DeprecationWarning,
    #     )

    # @property
    # def validity_filters(self):
    #     """The FilterList for the array's validity

    #     :rtype: tiledb.FilterList
    #     :raises: :py:exc:`tiledb.TileDBError`
    #     """
    #     cdef tiledb_filter_list_t* validity_list_ptr = NULL
    #     check_error(self.ctx,
    #         tiledb_array_schema_get_validity_filter_list(
    #             self.ctx.ptr, self.ptr, &validity_list_ptr))
    #     return FilterList(
    #         PyCapsule_New(validity_list_ptr, "fl", NULL),
    #             is_capsule=True, ctx=self.ctx)

    @property
    def domain(self):
        """The Domain associated with the array.

        :rtype: tiledb.Domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return Domain(_lt_obj=self._domain)

    @property
    def nattr(self):
        """The number of array attributes.

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._nattr

    @property
    def ndim(self):
        """The number of array domain dimensions.

        :rtype: int
        """
        return self._domain.ndim

    @property
    def shape(self):
        """The array's shape

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain
        """
        return self.domain.shape

    # @property
    # def version(self):
    #     """The array's scehma version.

    #     :rtype: int
    #     :raises :py:exc:`tiledb.TileDBError`
    #     """
    #     cdef uint32_t version
    #     check_error(self.ctx,
    #                 tiledb_array_schema_get_version(
    #                     self.ctx.ptr, self.ptr, &version))
    #     return version

    # def _make_invalid(self):
    #     """This is a helper function for testing schema.check: resets schema
    #     in order to make the schema invalid."""
    #     cdef tiledb_array_schema_t* schema_ptr = self.ptr
    #     tiledb_array_schema_free(&schema_ptr)
    #     check_error(self.ctx,
    #                 tiledb_array_schema_alloc(self.ctx.ptr, TILEDB_DENSE, &self.ptr))

    # def _needs_var_buffer(self, unicode name):
    #     """
    #     Returns true if the given attribute or dimension is var-sized
    #     :param name:
    #     :rtype: bool
    #     """
    #     if self.has_attr(name):
    #         return self.attr(name).isvar
    #     elif self.domain.has_dim(name):
    #         return self.domain.dim(name).isvar
    #     else:
    #         raise ValueError(f"Requested name '{name}' is not an attribute or dimension")

    def attr(self, key):
        """Returns an Attr instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: tiledb.Attr
        :return: The ArraySchema attribute at index or with the given name (label)
        :raises TypeError: invalid key type

        """
        if isinstance(key, str):
            return Attr(_lt_obj=self._attr(key))
        elif isinstance(key, numbers.Integral):
            return Attr(_lt_obj=self._attr(int(key)))
        raise TypeError(
            "attr indices must be a string name, "
            "or an integer index, not {0!r}".format(type(key))
        )

    # def has_attr(self, name):
    #     """Returns true if the given name is an Attribute of the ArraySchema

    #     :param name: attribute name
    #     :rtype: boolean
    #     """
    #     cdef:
    #         int32_t has_attr = 0
    #         int32_t rc = TILEDB_OK
    #         bytes bname = name.encode("UTF-8")

    #     rc = tiledb_array_schema_has_attribute(
    #         self.ctx.ptr,
    #         self.ptr,
    #         bname,
    #         &has_attr
    #     )
    #     if rc != TILEDB_OK:
    #         _raise_ctx_err(self.ctx.ptr, rc)

    #     return bool(has_attr)

    # def attr_or_dim_dtype(self, unicode name):
    #     if self.has_attr(name):
    #         dtype = self.attr(name).dtype
    #     elif self.domain.has_dim(name):
    #         dtype = self.domain.dim(name).dtype
    #     else:
    #         raise TileDBError(f"Unknown attribute or dimension ('{name}')")

    #     if dtype.itemsize == 0:
    #         # special handling for flexible numpy dtypes: change itemsize from 0 to 1
    #         dtype = np.dtype((dtype, 1))
    #     return dtype

    # def dump(self):
    #     """Dumps a string representation of the array object to standard output (stdout)"""
    #     check_error(self.ctx,
    #                 tiledb_array_schema_dump(self.ctx.ptr, self.ptr, stdout))
    #     print("\n")
    #     return

    # def __repr__(self):
    #     # TODO support/use __qualname__
    #     output = io.StringIO()
    #     output.write("ArraySchema(\n")
    #     output.write("  domain=Domain(*[\n")
    #     for i in range(self.domain.ndim):
    #         output.write(f"    {repr(self.domain.dim(i))},\n")
    #     output.write("  ]),\n")
    #     output.write("  attrs=[\n")
    #     for i in range(self.nattr):
    #         output.write(f"    {repr(self.attr(i))},\n")
    #     output.write("  ],\n")
    #     output.write(
    #         f"  cell_order='{self.cell_order}',\n"
    #         f"  tile_order={repr(self.tile_order)},\n"
    #     )
    #     output.write(f"  capacity={self.capacity},\n")
    #     output.write(f"  sparse={self.sparse},\n")
    #     if self.sparse:
    #         output.write(f"  allows_duplicates={self.allows_duplicates},\n")

    #     output.write(")\n")

    #     return output.getvalue()

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")

        output.write("<tr><th>Domain</th></tr>")
        output.write(f"<tr><td>{self.domain._repr_html_()}</td></tr>")

        output.write("<tr><th>Attributes</th></tr>")
        output.write("<tr>")
        output.write("<td>")
        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>Is Nullable</th>")
        output.write("<th>Filters</th>")
        output.write("</tr>")
        for i in range(self.nattr):
            output.write(f"{self.attr(i)._repr_html_row_only_()}")
        output.write("</table>")
        output.write("</td>")
        output.write("</tr>")

        output.write("<tr><th>Cell Order</th></tr>")
        output.write(f"<tr><td>{self.cell_order}</td></tr>")

        output.write("<tr><th>Tile Order</th></tr>")
        output.write(f"<tr><td>{self.tile_order}</td></tr>")

        output.write("<tr><th>Capacity</th></tr>")
        output.write(f"<tr><td>{self.capacity}</td></tr>")

        output.write("<tr><th>Sparse</th></tr>")
        output.write(f"<tr><td>{self.sparse}</td></tr>")

        if self.sparse:
            output.write("<tr><th>Allows DuplicatesK/th></tr>")
            output.write(f"<tr><td>{self.allows_duplicates}</td></tr>")

        output.write("</table>")

        return output.getvalue()
