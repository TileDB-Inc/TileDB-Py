"""Minimal Pandas ExtensionDtype and ExtensionArray for representing ragged arrays"""

import re

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


@pd.api.extensions.register_extension_dtype
class RaggedDtype(pd.api.extensions.ExtensionDtype):
    type = np.ndarray
    na_value = None

    def __init__(self, subtype=np.float64):
        self.subtype = np.dtype(subtype)

    @property
    def name(self):
        return f"Ragged[{self.subtype}]"

    @classmethod
    def construct_array_type(cls):
        return RaggedArray

    @classmethod
    def construct_from_string(cls, string):
        if string.lower() == "ragged":
            return cls()
        match = re.match(r"^ragged\[(\w+)\]$", string, re.IGNORECASE)
        if match:
            return cls(match.group(1))
        raise TypeError(f"Cannot construct a 'RaggedDtype' from '{string}'")


class RaggedArray(pd.api.extensions.ExtensionArray):
    def __init__(self, arrays, dtype):
        assert isinstance(dtype, RaggedDtype)
        self._dtype = dtype
        self._flat_arrays = [np.asarray(array, dtype=dtype.subtype) for array in arrays]

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars, dtype)

    def __len__(self):
        return len(self._flat_arrays)

    def __getitem__(self, i):
        return self._flat_arrays[i]

    @property
    def dtype(self):
        return self._dtype

    def copy(self):
        return type(self)(self._flat_arrays, self._dtype)
