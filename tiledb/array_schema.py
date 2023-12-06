import io
import numbers
import warnings
from typing import Sequence, Tuple, Union

import numpy as np

import tiledb.cc as lt

from .attribute import Attr
from .ctx import Ctx, CtxMixin, default_ctx
from .dimension_label import DimLabel
from .domain import Domain
from .filter import Filter, FilterList


class ArraySchema(CtxMixin, lt.ArraySchema):
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
    :param dim_labels: dict(dim_index, dict(dim_name, tiledb.DimSchema))
    :param tiledb.Ctx ctx: A TileDB Context
    :raises: :py:exc:`tiledb.TileDBError`

    """

    def __init__(
        self,
        domain: Domain = None,
        attrs: Sequence[Attr] = (),
        cell_order: str = "row-major",
        tile_order: str = "row-major",
        capacity: int = 0,
        coords_filters: Union[FilterList, Sequence[Filter]] = None,
        offsets_filters: Union[FilterList, Sequence[Filter]] = None,
        validity_filters: Union[FilterList, Sequence[Filter]] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
        dim_labels={},
        enums=None,
        ctx: Ctx = None,
    ):
        super().__init__(ctx, lt.ArrayType.SPARSE if sparse else lt.ArrayType.DENSE)

        if enums is not None:
            for enum_name in enums:
                self._add_enumeration(self._ctx, enum_name)

        if attrs is not None:
            for att in attrs:
                if not isinstance(att, Attr):
                    raise TypeError(
                        "Cannot create schema with non-Attr value for 'attrs' argument"
                    )
                self._add_attr(att)

        self._cell_order = _string_to_tiledb_order.get(cell_order)
        if self._cell_order is None:
            raise ValueError(f"unknown tiledb layout: {cell_order}")

        self._tile_order = _string_to_tiledb_order.get(tile_order)
        if self._tile_order is None:
            raise ValueError(f"unknown tiledb layout: {tile_order}")

        if capacity > 0:
            self._capacity = capacity

        if coords_filters is not None:
            warnings.warn(
                "coords_filters is deprecated; set the FilterList for each dimension",
                DeprecationWarning,
            )

            self._coords_filters = FilterList()

            dims_with_coords_filters = []
            for dim in domain:
                dim._filters = FilterList(coords_filters)
                dims_with_coords_filters.append(dim)
            domain = Domain(dims_with_coords_filters)

        if domain is not None:
            self._domain = domain

        if offsets_filters is not None:
            self._offsets_filters = FilterList(offsets_filters)

        if validity_filters is not None:
            self._validity_filters = FilterList(validity_filters)

        self._allows_dups = allows_duplicates

        for dim_index, labels_on_dim in dim_labels.items():
            for label_name, label_schema in labels_on_dim.items():
                self._add_dim_label(self._ctx, label_name, dim_index, label_schema)

        self._check()

    @classmethod
    def load(cls, uri, ctx: Ctx = None, key: str = None):
        if not ctx:
            ctx = default_ctx()

        args = [ctx, uri]
        if key is not None:
            args.extend((lt.EncryptionType.AES_256_GCM, key))

        return cls.from_pybind11(ctx, lt.ArraySchema(*args))

    @classmethod
    def from_file(cls, uri: str = None, ctx: Ctx = None):
        """Create an ArraySchema for a Filestore Array from a given file.
        If a uri is not given, then create a default schema."""
        schema = lt.Filestore._schema_create(ctx or default_ctx(), uri)
        return cls.from_pybind11(ctx, schema)

    def __eq__(self, other):
        """Instance is equal to another ArraySchema"""
        if not isinstance(other, ArraySchema):
            return False
        if not (
            self.sparse == other.sparse
            and self.cell_order == other.cell_order
            and self.tile_order == other.tile_order
            and self.capacity == other.capacity
            and self.coords_filters == other.coords_filters
            and self.offsets_filters == other.offsets_filters
            and self.validity_filters == other.validity_filters
            and self.nattr == other.nattr
            and self.domain == other.domain
        ):
            return False
        if self.sparse and self.allows_duplicates != other.allows_duplicates:
            return False
        for i in range(self.nattr):
            if self.attr(i) != other.attr(i):
                return False
        return True

    def __len__(self):
        """Returns the number of Attributes in the ArraySchema"""
        return self._nattr

    def __iter__(self):
        """Returns a generator object that iterates over the ArraySchema's Attribute objects"""
        return (self.attr(i) for i in range(self.nattr))

    @property
    def ctx(self) -> Ctx:
        """The array schema's context

        :rtype: tiledb.Ctx
        """
        return self._ctx

    def check(self) -> bool:
        """Checks the correctness of the array schema

        :rtype: None
        :raises: :py:exc:`tiledb.TileDBError` if invalid
        """
        return self._check()

    @property
    def sparse(self) -> bool:
        """True if the array is a sparse array representation

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._array_type == lt.ArrayType.SPARSE

    @property
    def allows_duplicates(self) -> bool:
        """Returns True if the (sparse) array allows duplicates."""

        if not self.sparse:
            raise lt.TileDBError(
                "ArraySchema.allows_duplicates does not apply to dense arrays"
            )

        return self._allows_dups

    @property
    def capacity(self) -> int:
        """The array capacity

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._capacity

    @property
    def cell_order(self) -> str:
        """The cell order layout of the array.

        :rtype: str
        """
        return _tiledb_order_to_string[self._cell_order]

    @property
    def tile_order(self) -> str:
        """The tile order layout of the array.

        :rtype: str
        :raises: :py:exc:`tiledb.TileDBError`

        """
        layout_string = _tiledb_order_to_string[self._tile_order]
        return layout_string if self.cell_order != "hilbert" else None

    @property
    def offsets_filters(self) -> FilterList:
        """The FilterList for the array's variable-length attribute offsets

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        return FilterList.from_pybind11(self._ctx, self._offsets_filters)

    @property
    def coords_filters(self) -> FilterList:
        """The FilterList for the array's coordinates

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        return FilterList.from_pybind11(self._ctx, self._coords_filters)

    @coords_filters.setter
    def coords_filters(self, value):
        warnings.warn(
            "coords_filters is deprecated; set the FilterList for each dimension",
            DeprecationWarning,
        )

    @property
    def validity_filters(self) -> FilterList:
        """The FilterList for the array's validity

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        return FilterList.from_pybind11(self._ctx, self._validity_filters)

    @property
    def domain(self) -> Domain:
        """The Domain associated with the array.

        :rtype: tiledb.Domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return Domain.from_pybind11(self._ctx, self._domain)

    @property
    def nattr(self) -> int:
        """The number of array attributes.

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self._nattr

    @property
    def ndim(self) -> int:
        """The number of array domain dimensions.

        :rtype: int
        """
        return self.domain.ndim

    @property
    def shape(self) -> Tuple[np.dtype, np.dtype]:
        """The array's shape

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain
        """
        return self.domain.shape

    @property
    def version(self) -> int:
        """The array's schema (storage) version.

        :rtype: int
        :raises :py:exc:`tiledb.TileDBError`
        """
        return self._version

    def _needs_var_buffer(self, name: str) -> bool:
        """
        Returns true if the given attribute or dimension is var-sized
        :param name:
        :rtype: bool
        """
        if self.has_attr(name):
            return self.attr(name).isvar
        elif self.domain.has_dim(name):
            return self.domain.dim(name).isvar
        else:
            raise ValueError(
                f"Requested name '{name}' is not an attribute or dimension"
            )

    def attr(self, key: Union[str, int]) -> Attr:
        """Returns an Attr instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: tiledb.Attr
        :return: The ArraySchema attribute at index or with the given name (label)
        :raises TypeError: invalid key type

        """
        if isinstance(key, str):
            return Attr.from_pybind11(self._ctx, self._attr(key))
        elif isinstance(key, numbers.Integral):
            return Attr.from_pybind11(self._ctx, self._attr(int(key)))
        raise TypeError(
            "attr indices must be a string name, "
            "or an integer index, not {0!r}".format(type(key))
        )

    def dim_label(self, name: str) -> DimLabel:
        """Returns a TileDB DimensionLabel given the label name

        :param name: name of the dimensin label
        :return: The dimension label associated with the given name
        """
        return DimLabel.from_pybind11(self._ctx, self._dim_label(self._ctx, name))

    def has_attr(self, name: str) -> bool:
        """Returns true if the given name is an Attribute of the ArraySchema

        :param name: attribute name
        :rtype: boolean
        """
        return self._has_attribute(name)

    def has_dim_label(self, name: str) -> bool:
        """Returns true if the given name is a DimensionLabel of the ArraySchema

        Note: If using an version of libtiledb that does not support dimension labels
        this will return false.

        :param name: dimension label name
        :rtype: boolean
        """
        return self._has_dim_label(self._ctx, name)

    def attr_or_dim_dtype(self, name: str) -> bool:
        if self.has_attr(name):
            dtype = self.attr(name).dtype
        elif self.domain.has_dim(name):
            dtype = self.domain.dim(name).dtype
        else:
            raise lt.TileDBError(f"Unknown attribute or dimension ('{name}')")

        if dtype.itemsize == 0:
            # special handling for flexible numpy dtypes: change itemsize from 0 to 1
            dtype = np.dtype((dtype, 1))
        return dtype

    def dump(self):
        """Dumps a string representation of the array object to standard output (stdout)"""
        print(self._dump(), "\n")

    def __repr__(self):
        # TODO support/use __qualname__
        output = io.StringIO()
        output.write("ArraySchema(\n")
        output.write("  domain=Domain(*[\n")
        for i in range(self.domain.ndim):
            output.write(f"    {repr(self.domain.dim(i))},\n")
        output.write("  ]),\n")
        output.write("  attrs=[\n")
        for i in range(self.nattr):
            output.write(f"    {repr(self.attr(i))},\n")
        output.write("  ],\n")
        output.write(
            f"  cell_order='{self.cell_order}',\n"
            f"  tile_order={repr(self.tile_order)},\n"
        )
        if self.sparse:
            output.write(f"  capacity={self.capacity},\n")
        output.write(f"  sparse={self.sparse},\n")
        if self.sparse:
            output.write(f"  allows_duplicates={self.allows_duplicates},\n")

        output.write(")\n")

        return output.getvalue()

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

        if self.sparse:
            output.write("<tr><th>Capacity</th></tr>")
            output.write(f"<tr><td>{self.capacity}</td></tr>")

        output.write("<tr><th>Sparse</th></tr>")
        output.write(f"<tr><td>{self.sparse}</td></tr>")

        if self.sparse:
            output.write("<tr><th>Allows DuplicatesK/th></tr>")
            output.write(f"<tr><td>{self.allows_duplicates}</td></tr>")

        output.write("</table>")

        return output.getvalue()


_tiledb_order_to_string = {
    lt.LayoutType.ROW_MAJOR: "row-major",
    lt.LayoutType.COL_MAJOR: "col-major",
    lt.LayoutType.GLOBAL_ORDER: "global",
    lt.LayoutType.UNORDERED: "unordered",
    lt.LayoutType.HILBERT: "hilbert",
}

_string_to_tiledb_order = {v: k for k, v in _tiledb_order_to_string.items()}
_string_to_tiledb_order.update(
    {
        "C": lt.LayoutType.ROW_MAJOR,
        "R": lt.LayoutType.COL_MAJOR,
        "H": lt.LayoutType.HILBERT,
        "U": lt.LayoutType.UNORDERED,
        None: lt.LayoutType.ROW_MAJOR,  # default (fixed in SC-27374)
    }
)
