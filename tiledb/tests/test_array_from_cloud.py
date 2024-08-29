import tiledb
import os
import pytest

tiledb_token = os.getenv('TILEDB_TOKEN')
tiledb_path = os.getenv('TILEDB_PATH')
s3_path = os.getenv('S3_PATH')

@pytest.mark.skipif(tiledb_token is None, reason="No TileDB token found. Please set TILEDB_TOKEN env variable.")
class ArrayCloudTest:
    def test_open_array_from_cloud(self):
        config = tiledb.Config({"rest.token": tiledb_token})
        ctx = tiledb.Ctx(config=config)

        a = tiledb.open(tiledb_path, ctx=ctx)
        assert a.schema == tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="image_id", domain=(0, 18446744073709541615), tile=10, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
                tiledb.Dim(name="d1", domain=(0, 223), tile=224, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
                tiledb.Dim(name="d2", domain=(0, 223), tile=224, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
                tiledb.Dim(name="d3", domain=(0, 2), tile=2, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
            ),
            attrs=[
                tiledb.Attr(name="value", dtype="uint8", var=False, nullable=False, enum_label=None),
            ],
            cell_order="row-major",
            tile_order="row-major",
            sparse=False,
        )

        b = tiledb.open(s3_path, ctx=ctx)
        assert b.schema == tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="__dim_0", domain=(0, 3), tile=4, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
                tiledb.Dim(name="__dim_1", domain=(0, 3), tile=4, dtype="uint64", filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])),
            ),
            attrs=[
                tiledb.Attr(name="", dtype="float64", var=False, nullable=False, enum_label=None),
            ],
            cell_order="row-major",
            tile_order="row-major",
            sparse=False,
        )
