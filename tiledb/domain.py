from typing import TYPE_CHECKING

import tiledb.cc as lt
from .ctx import default_ctx
from .dimension import Dim

if TYPE_CHECKING:
    from .libtiledb import Ctx


class Domain(lt.Domain):
    """
    Represents a TileDB domain.
    """

    def __init__(self, *dims: Dim, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        _cctx = lt.Context(self._ctx, False)

        super().__init__(_cctx)

        for d in dims:
            self._add_dim(lt.Dimension(_cctx, d.__capsule__()))

    # def __repr__(self):
    #     dims = ",\n       ".join([repr(self.dim(i)) for i in range(self.ndim)])
    #     return "Domain({0!s})".format(dims)

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
        return self._ndim

    def __iter__(self):
        """Returns a generator object that iterates over the domain's dimension objects"""
        return (Dim(self._dim(i)) for i in range(self.ndim))

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

    @property
    def ndim(self):
        """The number of dimensions of the domain.

        :rtype: int

        """
        return self._ndim

    @property
    def dtype(self):
        """The numpy dtype of the domain's dimension type.

        :rtype: numpy.dtype

        """
        return self._numpy_dtype

    @property
    def shape(self):
        """The domain's shape, valid only for integer domains.

        :rtype: tuple
        :raises TypeError: floating point (inexact) domain

        """
        return tuple(dim.shape[0] for dim in self)

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

    def dim(self, dim_id):
        """Returns a Dim object from the domain given the dimension's index or name.

        :param dim_d: dimension index (int) or name (str)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return Dim(self._dim(dim_id))

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

    def dump(self):
        """Dumps a string representation of the domain object to standard output (STDOUT)"""
        self._dump()
