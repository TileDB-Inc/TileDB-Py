import io
import numpy as np
from typing import Any, Tuple, TYPE_CHECKING

import tiledb.cc as lt
from .ctx import default_ctx
from .util import dtype_to_tiledb, numpy_dtype

if TYPE_CHECKING:
    from .libtiledb import Ctx


class DimLabel(lt.DimensionLabel):
    """
    Represents a TileDB dimension label.
    """

    def __init__(
        self,
        ctx: "Ctx" = None,
        _lt_obj: lt.DimensionLabel = None,
        _capsule: "PyCapsule" = None,
    ):
        self._ctx = ctx or default_ctx()
        if _capsule is not None:
            return super().__init__(self._ctx, _capsule)
        if _lt_obj is not None:
            return super().__init__(_lt_obj)
        raise ValueError("either _lt_obj or _capsule must be provided")

    def __repr__(self) -> str:
        dtype = "ascii" if self.isascii else self.dtype
        return (
            f"DimLabel(name={self.name}, dtype='{dtype!s}', "
            f"var={self.isvar!s}, uri={self.uri})"
        )

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>URI</th>")
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
        output.write(f"<td>{self.uri}</td>")
        output.write("</tr>")

        return output.getvalue()

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype representation of the label type.

        :rtype: numpy.dtype

        """
        return np.dtype(numpy_dtype(self._tiledb_label_dtype))

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
    def name(self) -> str:
        """The name of the dimension label.

        :rtype: str

        """
        return self._name

    @property
    def uri(self) -> str:
        """The URI of the array containing the dimension label data.

        :rtype: str

        """
        return self._uri
