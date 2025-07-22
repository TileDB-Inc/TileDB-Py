# profile.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2025 TileDB, Inc.
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

# This example demonstrates how to create, save, and use profiles in TileDB.
# It also shows how to remove profiles when they are no longer needed.

import datetime
import os
import random
import string

import tiledb

tiledb_token = os.getenv("TILEDB_TOKEN")
tiledb_namespace = os.getenv("TILEDB_NAMESPACE")
s3_bucket = os.getenv("S3_BUCKET")


def create_and_save_profile():
    p1 = tiledb.Profile("my_profile_name")
    p1["rest.token"] = tiledb_token
    p1.save()


def use_profile():
    # Create a config object. This will use the default profile.
    cfg = tiledb.Config({"profile_name": "my_profile_name"})
    print("rest.token:", cfg["rest.token"])

    # Use on of the profile to create a context.
    ctx = tiledb.Ctx(cfg)

    # Use the context to create a new array. The REST credentials from the profile will be used.
    # Useful to include the datetime in the array name to handle multiple consecutive runs of the test.
    # Random letters are added to the end to ensure that conflicts are avoided, especially in CI environments where multiple tests may run in parallel.
    array_name = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "-"
        + "".join(random.choice(string.ascii_letters) for _ in range(5))
    )
    uri = f"tiledb://{tiledb_namespace}/s3://{s3_bucket}/{array_name}"
    dom = tiledb.Domain(tiledb.Dim(name="d", domain=(1, 10), tile=5, dtype="int32"))
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="float64")]
    )
    tiledb.Array.create(uri, schema, ctx=ctx)


def remove_profile():
    # Remove the default profile
    tiledb.Profile.remove("my_profile_name")


if __name__ == "__main__":
    create_and_save_profile()
    use_profile()
    remove_profile()
