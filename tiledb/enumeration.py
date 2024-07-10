from __future__ import annotations

import io
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin
from .datatypes import DataType


class Enumeration(CtxMixin, lt.Enumeration):
    """
    Represents a TileDB Enumeration.
    """

    def __init__(
        self,
        name: str,
        ordered: bool,
        values: Optional[Sequence[Any]] = None,
        dtype: Optional[np.dtype] = None,
        ctx: Optional[Ctx] = None,
    ):
        """Class representing the TileDB Enumeration.

        :param name: The name of the to-be created Enumeration
        :type name: str
        :param ordered: Whether or not to consider this enumeration ordered
        :type ordered: bool
        :param values: A Numpy array of values for this enumeration
        :type values: np.array
        :param dtype: The Numpy data type for this enumeration
        :type dtype: np.dtype
        :param ctx: A TileDB context
        :type ctx: tiledb.Ctx
        """
        if values is None or len(values) == 0:
            if dtype is None:
                raise ValueError("dtype must be provided for empty enumeration")
            super().__init__(ctx, name, np.dtype(dtype), ordered)

        values = np.array(values)
        if np.dtype(values.dtype).kind in "US":
            dtype = (
                lt.DataType.STRING_UTF8
                if values.dtype.kind == "U"
                else lt.DataType.STRING_ASCII
            )
            super().__init__(ctx, name, values, ordered, dtype)
        else:
            super().__init__(ctx, name, ordered, values, np.array([]))

    @property
    def name(self) -> str:
        """The enumeration label string.

        :rtype: str
        """
        return super().name

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype representation of the enumeration type.

        :rtype: numpy.dtype
        """
        return DataType.from_tiledb(super().type).np_dtype

    @property
    def cell_val_num(self) -> int:
        """The enumeration's cell value number.

        :rtype: int
        """
        return super().cell_val_num

    @property
    def ordered(self) -> bool:
        """True if the enumeration is ordered.

        :rtype: bool
        """
        return super().ordered

    def values(self) -> NDArray:
        """The values of the enumeration.

        :rtype: NDArray
        """
        if self.dtype.kind == "U":
            return np.array(super().str_values(), dtype=np.str_)
        elif self.dtype.kind == "S":
            return np.array(super().str_values(), dtype=np.bytes_)
        else:
            return np.array(super().values(), dtype=self.dtype)

    def extend(self, values: Sequence[Any]) -> Enumeration:
        """Add additional values to the enumeration.

        :rtype: Enumeration
        """
        values = np.array(values)
        if self.dtype.kind in "US" and values.dtype.kind not in "US":
            raise lt.TileDBError("Passed in enumeration must be string type")

        if np.issubdtype(self.dtype, np.integer) and not np.issubdtype(
            values.dtype, np.integer
        ):
            raise lt.TileDBError("Passed in enumeration must be integer type")

        return Enumeration.from_pybind11(self._ctx, super().extend(values))

    def __eq__(self, other):
        if not isinstance(other, Enumeration):
            return False

        return all(
            [
                self.name == other.name,
                self.dtype == other.dtype,
                self.cell_val_num == other.cell_val_num,
                self.ordered == other.ordered,
                np.array_equal(self.values(), other.values()),
            ]
        )

    def __repr__(self):
        # use safe repr if pybind11 constructor failed
        if self._ctx is None:
            return object.__repr__(self)

        return f"Enumeration(name='{self.name}', dtype={self.dtype}, dtype_name='{self.dtype.name}', cell_val_num={self.cell_val_num}, ordered={self.ordered}, values={list(self.values())})"

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Ordered</th>")
        output.write("</tr>")
        output.write(f"{self._repr_html_row_only_()}")
        output.write("</table>")

        return output.getvalue()

    def _repr_html_row_only_(self):
        output = io.StringIO()

        output.write("<tr>")
        output.write(f"<td>{self.name}</td>")
        output.write(f"<td>{self.dtype}</td>")
        output.write(f"<td>{self.cell_val_num}</td>")
        output.write(f"<td>{self.ordered}</td>")
        output.write("</tr>")

        return output.getvalue()
