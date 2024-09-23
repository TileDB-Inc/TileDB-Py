import numpy as np

import tiledb
import tiledb.cc as lt


class PackedBuffer:
    def __init__(self, data, tdbtype, value_num):
        self.data = data
        self.tdbtype = tdbtype
        self.value_num = value_num


def pack_metadata_val(value) -> PackedBuffer:
    if isinstance(value, bytes):
        return PackedBuffer(value, lt.DataType.BLOB, len(value))

    if isinstance(value, str):
        value = value.encode("UTF-8")
        return PackedBuffer(value, lt.DataType.STRING_UTF8, len(value))

    if not isinstance(value, (list, tuple)):
        value = (value,)

    if not value:
        # special case for empty values
        return PackedBuffer(b"", lt.DataType.INT32, 0)

    val0 = value[0]
    if not isinstance(val0, (int, float)):
        raise TypeError(f"Unsupported item type '{type(val0)}'")

    tiledb_type = lt.DataType.INT64 if isinstance(val0, int) else lt.DataType.FLOAT64
    python_type = int if isinstance(val0, int) else float
    numpy_dtype = np.int64 if isinstance(val0, int) else np.float64
    data_view = memoryview(
        bytearray(len(value) * tiledb.main.datatype_size(tiledb_type))
    )

    for i, val in enumerate(value):
        if not isinstance(val, python_type):
            raise TypeError(f"Mixed-type sequences are not supported: {value}")
        data_view[i * 8 : (i + 1) * 8] = numpy_dtype(val).tobytes()

    return PackedBuffer(bytes(data_view), tiledb_type, len(value))
