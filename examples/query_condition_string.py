# query_condition_string.py
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

# This example creates an array with one string-typed attribute,
# writes sample data to the array, and then prints out a filtered
# dataframe using the TileDB QueryCondition feature.

import string

import numpy as np

import tiledb


def create_array(path):
    dom = tiledb.Domain(tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint32))
    attrs = [tiledb.Attr(name="ascii_attr", dtype="ascii", var=True)]

    schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)

    tiledb.SparseArray.create(path, schema, overwrite=True)

    # create array of strings from a to a..j
    attr_data = np.array([string.ascii_lowercase[0:n] for n in range(1, 11)], dtype="O")

    with tiledb.open(path, "w") as arr:
        arr[np.arange(1, 11)] = {"ascii_attr": attr_data}


def read_array(path, cond):
    with tiledb.open(path) as arr:
        print("QueryCondition is: ", cond)
        res = arr.query(cond=cond)[:]
        return res


uri = "query_condition_string"

create_array(uri)
filtered_df1 = read_array(uri, "ascii_attr == 'abcd'")
print("  result: ", filtered_df1)
filtered_df2 = read_array(uri, "ascii_attr > 'abc'")
print("  result: ", filtered_df2)
