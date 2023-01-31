import io
from typing import Optional

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .dimension import Dim
from .util import numpy_dtype


class Domain(CtxMixin, lt.Domain):
    """
    Represents a TileDB domain.
    """

    def __init__(self, *dims: Dim, ctx: Optional[Ctx] = None):
        """Class representing the domain of a TileDB Array.

        :param *dims*: one or more tiledb.Dim objects up to the Domain's ndim
        :param ctx: A TileDB Context
        :raises TypeError: All dimensions must have the same dtype
        :raises tiledb.TileDBError:
        """
        super().__init__(ctx)

        # support passing a list of dims without splatting
        if len(dims) == 1 and isinstance(dims[0], list):
            dims = dims[0]

        if len(dims) == 0:
            raise lt.TileDBError("Domain must have ndim >= 1")

        if len(dims) > 1:
            if all(dim.name == "__dim_0" for dim in dims):

                def clone_dim_with_name(dim, name):
                    return Dim(
                        name=name,
                        domain=dim.domain,
                        tile=dim.tile,
                        dtype=dim.dtype,
                        ctx=dim._ctx,
                    )

                # rename anonymous dimensions sequentially
                dims = [
                    clone_dim_with_name(dims[i], name=f"__dim_{i}")
                    for i in range(len(dims))
                ]
            elif any(dim.name.startswith("__dim_0") for dim in dims[1:]):
                raise lt.TileDBError(
                    "Mixed dimension naming: dimensions must be either all anonymous or all named."
                )

        for d in dims:
            if not isinstance(d, Dim):
                raise TypeError(
                    "Cannot create Domain with non-Dim value for 'dims' argument"
                )
            self._add_dim(d)

    def __repr__(self):
        dims = ",\n       ".join(repr(self.dim(i)) for i in range(self.ndim))
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
        return self._ndim

    def __iter__(self):
        """Returns a generator object that iterates over the domain's dimension objects"""
        return (Dim.from_pybind11(self._ctx, self._dim(i)) for i in range(self.ndim))

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
        return np.dtype(numpy_dtype(self._tiledb_dtype))

    @property
    def shape(self):
        """The domain's shape, valid only for integer domains.

        :rtype: tuple
        :raises TypeError: floating point (inexact) domain

        """
        return tuple(dim.shape[0] for dim in self)

    @property
    def size(self):
        """The domain's size (number of cells), valid only for integer domains.

        :rtype: int
        :raises TypeError: floating point (inexact) domain

        """
        if self._tiledb_dtype not in (
            lt.DataType.UINT8,
            lt.DataType.INT8,
            lt.DataType.UINT16,
            lt.DataType.INT16,
            lt.DataType.UINT32,
            lt.DataType.INT32,
            lt.DataType.UINT64,
            lt.DataType.INT64,
        ):
            raise TypeError("shape valid only for integer domains")
        return np.product(self.shape)

    def _is_homogeneous(self):
        dtype0 = self.dim(0).dtype
        return all(self.dim(i).dtype == dtype0 for i in range(1, self.ndim))

    @property
    def homogeneous(self):
        """Returns True if the domain's dimension types are homogeneous."""
        return self._is_homogeneous()

    def dim(self, dim_id):
        """Returns a Dim object from the domain given the dimension's index or name.

        :param dim_d: dimension index (int) or name (str)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not isinstance(dim_id, (int, str)):
            raise ValueError(
                f"Unsupported dim identifier: '{dim_id!r}' (expected int or str)"
            )
        return Dim.from_pybind11(self._ctx, self._dim(dim_id))

    def has_dim(self, name):
        """
        Returns true if the Domain has a Dimension with the given name

        :param name: name of Dimension
        :rtype: bool
        :return:
        """
        return self._has_dim(name)

    def dump(self):
        """Dumps a string representation of the domain object to standard output (STDOUT)"""
        self._dump()
