import datetime
import os
import random
import string

import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase

tiledb_token = os.getenv("TILEDB_TOKEN")
tiledb_namespace = os.getenv("TILEDB_NAMESPACE")
s3_bucket = os.getenv("S3_BUCKET")


@pytest.mark.skipif(
    tiledb_token == None
    or tiledb_namespace == None
    or s3_bucket == None
    or tiledb_token == ""
    or tiledb_namespace == ""
    or s3_bucket == "",
    reason="No token was provided in a non-CI environment. Please set the TILEDB_TOKEN environment variable to run this test.",
)
class CloudTest(DiskTestCase):
    def test_save_and_open_array_from_cloud(self):
        config = tiledb.Config({"rest.token": tiledb_token})
        ctx = tiledb.Ctx(config=config)

        # Useful to include the datetime in the array name to handle multiple consecutive runs of the test.
        # Random letters are added to the end to ensure that conflicts are avoided, especially in CI environments where multiple tests may run in parallel.
        array_name = (
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + "-"
            + "".join(random.choice(string.ascii_letters) for _ in range(5))
        )
        uri = f"tiledb://{tiledb_namespace}/s3://{s3_bucket}/{array_name}"

        with tiledb.from_numpy(uri, np.random.rand(3, 2), ctx=ctx) as T:
            self.assertTrue(tiledb.array_exists(uri, ctx=ctx))
            self.assertTrue(
                T.schema
                == tiledb.ArraySchema(
                    domain=tiledb.Domain(
                        tiledb.Dim(
                            name="__dim_0",
                            domain=(0, 2),
                            tile=3,
                            dtype="uint64",
                            filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)]),
                        ),
                        tiledb.Dim(
                            name="__dim_1",
                            domain=(0, 1),
                            tile=2,
                            dtype="uint64",
                            filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)]),
                        ),
                    ),
                    attrs=[
                        tiledb.Attr(
                            name="",
                            dtype="float64",
                            var=False,
                            nullable=False,
                            enum_label=None,
                        ),
                    ],
                    cell_order="row-major",
                    tile_order="row-major",
                    sparse=False,
                )
            )

        tiledb.Array.delete_array(uri, ctx=ctx)

    def test_save_and_open_array_from_cloud_with_profile(self):
        # Create and save a profile with the TileDB token
        p = tiledb.Profile("my_profile_name")
        p["rest.token"] = tiledb_token
        p.save()

        # Create a config object. This will use the default profile.
        cfg = tiledb.Config({"profile_name": "my_profile_name"})
        assert cfg["rest.token"] == tiledb_token

        # Use one of the profiles to create a context.
        ctx = tiledb.Ctx(cfg)

        # Useful to include the datetime in the array name to handle multiple consecutive runs of the test.
        # Random letters are added to the end to ensure that conflicts are avoided, especially in CI
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

        data = np.random.rand(10)

        with tiledb.open(uri, "w", ctx=ctx) as A:
            A[:] = data

        self.assertTrue(tiledb.array_exists(uri, ctx=ctx))

        with tiledb.open(uri, "r", ctx=ctx) as A:
            np.testing.assert_array_equal(data, A[:]["a"])

        # Clean up
        tiledb.Array.delete_array(uri, ctx=ctx)
        tiledb.Profile.remove("my_profile_name")
