# query_condition_datetime.py
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

# This example creates an array with one datetime-typed attribute,
# writes sample data to the array, and then prints out a filtered
# dataframe using the TileDB QueryCondition feature to select on
# either equality or ranges of the generated attribute values.

import numpy as np
import pandas as pd

import tiledb

uri = "query_condition_datetime"

data = pd.DataFrame(
    np.sort(np.random.randint(438923600, 243892360000, 20, dtype=np.int64)).astype(
        "M8[ns]"
    ),
    columns=["dates"],
)
data.sort_values(by="dates")

tiledb.from_pandas(
    uri,
    data,
    column_types={"dates": "datetime64[ns]"},
)

with tiledb.open(uri) as A:
    # filter by exact match with the fifth cell
    search_date = data["dates"][5].to_numpy().astype(np.int64)
    result = A.query(cond=f"dates == {search_date}").df[:]

    print()
    print("Attribute dates matching index 5:")
    print(result)

    # filter values between cell index 3 and 8
    d1 = data["dates"].iloc[3].to_numpy().astype(np.int64)
    d2 = data["dates"].iloc[8].to_numpy().astype(np.int64)
    result2 = A.query(cond=f"dates > {d1} and dates < {d2}").df[:]

    print()
    print("Attribute dates where 'dates[3] < val < dates[8]'")
    print(result2)
