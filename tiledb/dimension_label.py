import io

import numpy as np

import tiledb.cc as lt

from .ctx import CtxMixin
from .data_order import DataOrder
from .datatypes import DataType


class DimLabel(CtxMixin, lt.DimensionLabel):
    """
    Represents a TileDB dimension label.
    """

    def __repr__(self) -> str:
        dtype = "ascii" if self.isascii else self.dtype
        return (
            f"DimLabel(name={self.name}, dtype='{dtype!s}', "
            f"var={self.isvar!s}, order={self.order!s})"
        )

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>Data Order</th>")
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
        output.write(f"<td>{self.order}</td>")
        output.write("</tr>")

        return output.getvalue()

    @property
    def dim_index(self) -> int:
        """Index of the dimension the labels are for.

        :rtype: int

        """
        return self._dim_index

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype representation of the label type.

        :rtype: numpy.dtype

        """
        return DataType.from_tiledb(self._tiledb_label_dtype).np_dtype

    @property
    def isvar(self) -> bool:
        """True if the labels are variable length.

        :rtype: bool

        """
        return self._label_ncell == lt.TILEDB_VAR_NUM()

    @property
    def isascii(self) -> bool:
        """True if the labels are variable length.

        :rtype: bool

        """
        return self._tiledb_label_dtype == lt.DataType.STRING_ASCII

    @property
    def label_attr_name(self) -> str:
        """Name of the attribute storing the label data.

        :rtype: str
        """
        return self._label_attr_name

    @property
    def name(self) -> str:
        """The name of the dimension label.

        :rtype: str

        """
        return self._name

    @property
    def order(self) -> str:
        """The order of the label data in the dimension label.

        :rtype: str
        """
        return DataOrder(self._tiledb_label_order).name

    @property
    def uri(self) -> str:
        """The URI of the array containing the dimension label data.

        :rtype: str

        """
        return self._uri
