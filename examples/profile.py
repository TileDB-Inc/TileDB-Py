# query_condition_sparse.py
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

import tiledb


def create_and_save_profiles():
    p1 = tiledb.Profile()
    p1["rest.token"] = "my_token"
    p1.save()

    p2 = tiledb.Profile("my_profile_name")
    p2["rest.server_address"] = "https://my.address"
    p2.save()


def use_profiles():
    # Create a config object. This will use the default profile.
    cfg = tiledb.Config()
    print("rest.token:", cfg["rest.token"])

    # Create a config object using a specific profile name.
    cfg_with_profile = tiledb.Config({"profile_name": "my_profile_name"})
    print("rest.server_address:", cfg_with_profile["rest.server_address"])

    # Use on of the profile to create a context.
    ctx = tiledb.Ctx(cfg_with_profile)

    # Use the context to create a new array. The REST credentials from the profile will be used.
    array_name = "tiledb://my_workspace/my_teamspace/my_array"
    dom = tiledb.Domain(tiledb.Dim(name="d", domain=(1, 10), tile=5, dtype="int32"))
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="float64")]
    )
    tiledb.Array.create(array_name, schema, ctx=ctx)


def remove_profiles():
    # Remove the default profile
    tiledb.Profile.remove()

    # Remove a specific profile by name
    tiledb.Profile.remove("my_profile_name")


if __name__ == "__main__":
    create_and_save_profiles()
    use_profiles()
    remove_profiles()
