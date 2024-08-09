# query_condition_dense.py
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

from pprint import pprint

import numpy as np

import tiledb

uri = "query_condition_dense"


def create_array(path):
    # create a dense array
    dom = tiledb.Domain(
        tiledb.Dim(name="coords", domain=(1, 10), tile=1, dtype=np.uint32)
    )
    attrs = [
        tiledb.Attr(name="attr1", dtype=np.uint64),
        tiledb.Attr(name="attr2", dtype=np.float64),
    ]
    schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
    tiledb.Array.create(path, schema, overwrite=True)

    # fill array with randomized values
    with tiledb.open(path, "w") as arr:
        rand = np.random.default_rng()
        arr[:] = {
            "attr1": rand.integers(low=0, high=10, size=10),
            "attr2": rand.random(size=10),
        }


def read_array(path):
    with tiledb.open(uri) as arr:
        print("--- without query condition:")
        print()
        pprint(arr[:])
        print()

    with tiledb.open(uri) as arr:
        qc = "(2 < attr1 < 6) and (attr2 < 0.5 or attr2 > 0.85)"
        print(f"--- with query condition {qc}:")

        print(f"--- the fill value for attr1 is {arr.attr('attr1').fill}")
        print(f"--- the fill value for attr2 is {arr.attr('attr2').fill}")

        print()
        res = arr.query(cond=qc)[:]
        pprint(res)


if __name__ == "__main__":
    """Example output for `python query_condition_dense.py`:

    --- without query condition:

    OrderedDict([('attr1', array([4, 0, 9, 7, 6, 0, 0, 5, 7, 5], dtype=uint64)),
                ('attr2',
                array([0.74476144, 0.47211544, 0.99054245, 0.36640416, 0.91699594,
        0.06216043, 0.58581863, 0.00505695, 0.7486192 , 0.87649422]))])

    --- with query condition (2 < attr1 < 6) and (attr2 < 0.5 or attr2 > 0.85):
    --- the fill value for attr1 is [18446744073709551615]
    --- the fill value for attr2 is [nan]

    OrderedDict([('attr1',
                array([18446744073709551615, 18446744073709551615, 18446744073709551615,
        18446744073709551615, 18446744073709551615, 18446744073709551615,
        18446744073709551615,                    5, 18446744073709551615,
                            5], dtype=uint64)),
                ('attr2',
                array([       nan,        nan,        nan,        nan,        nan,
                nan,        nan, 0.00505695,        nan, 0.87649422]))])
    """
    create_array(uri)
    read_array(uri)
