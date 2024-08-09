# writing_dense_rgb.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2021 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# DESCRIPTION
#
# Please see the TileDB documentation for more information:
#   https://docs.tiledb.com/main/how-to/arrays/writing-arrays/writing-in-dense-subarrays
#
# When run, this program will create a 2D+1 multi-component (eg RGB) dense array, write some
# data to it, and read the entire array data.

import numpy as np

import tiledb

img_shape = (100, 224, 224)
img_uri = "writing_dense_rgb"

image_data = np.random.randint(low=0, high=100, size=(*img_shape, 3), dtype=np.int32)


def create_array():
    domain = tiledb.Domain(
        tiledb.Dim(
            name="image_id", domain=(0, img_shape[0] - 1), tile=4, dtype=np.int32
        ),
        tiledb.Dim(
            name="x", domain=(0, img_shape[1] - 1), tile=img_shape[1], dtype=np.int32
        ),
        tiledb.Dim(
            name="y", domain=(0, img_shape[2] - 1), tile=img_shape[2], dtype=np.int32
        ),
    )

    # create multi-component attribute with three int32 components
    attr = tiledb.Attr(dtype=np.dtype("i4, i4, i4"))

    schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=[attr])

    tiledb.Array.create(img_uri, schema, overwrite=True)

    image_data_rgb = image_data.view(np.dtype("i4, i4, i4")).reshape(img_shape)

    with tiledb.open(img_uri, "w") as A:
        # write data to 1st image_id slot
        A[:] = image_data_rgb


def read_array():
    with tiledb.open(img_uri) as A:
        print(A[:].shape)


if __name__ == "__main__":
    create_array()
    read_array()
