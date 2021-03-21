"""Minimal Pandas ExtensionDtype and ExtensionArray for representing ragged arrays"""

import re

import numpy as np

from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)


@register_extension_dtype
class RaggedDtype(ExtensionDtype):

    type = np.ndarray
    na_value = None

    def __init__(self, dtype=np.float64):
        self._dtype = np.dtype(dtype)

    @property
    def subtype(self):
        return self._dtype

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


class RaggedArray(ExtensionArray):
    def __init__(self, arrays, dtype=None):
        if isinstance(dtype, RaggedDtype):
            self._dtype = dtype
            dtype = dtype.subtype
        else:
            self._dtype = RaggedDtype(dtype)

        self._flat_arrays = [np.asarray(array, dtype=dtype) for array in arrays]

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
