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
    os.getenv("CI") == None
    or tiledb_token == None
    or tiledb_namespace == None
    or s3_bucket == None,
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
