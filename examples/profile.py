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

# This example demonstrates how to create, save, use, and remove a Profile in TileDB.
# NOTE: This example is not intended to be run as a test (test_examples.py excludes it)
# since it requires a TileDB REST token.
# Profiles are getting checked in test_cloud.py.

import tiledb


def create_and_save_profile():
    p1 = tiledb.Profile("my_profile_name")
    p1["rest.token"] = "my_token"  # Replace with your actual TileDB token
    p1.save()


def use_profile():
    # Create a config object passing the desired profile name.
    cfg = tiledb.Config({"profile_name": "my_profile_name"})
    print("rest.token:", cfg["rest.token"])

    # Use the config to create a context.
    ctx = tiledb.Ctx(cfg)

    # By using the context to create a new array, the REST credentials from the profile will be used.

    uri = f"tiledb://<workspace>/<teamspace>/<array-name>"

    dom = tiledb.Domain(tiledb.Dim(name="d", domain=(1, 10), tile=5, dtype="int32"))
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="float64")]
    )
    # This will be a REST operation if the URI is set to a cloud location.
    tiledb.Array.create(uri, schema, ctx=ctx)


def remove_profile():
    # Remove the default profile
    tiledb.Profile.remove("my_profile_name")


if __name__ == "__main__":
    create_and_save_profile()
    use_profile()
    remove_profile()
