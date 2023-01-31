import shutil
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import tiledb
import tiledb.main as main


@pytest.mark.skipif(
    not main.test_webp_filter.webp_filter_exists(),
    reason="Can't create WebP filter; built with TILEDB_WEBP=OFF",
)
@pytest.mark.parametrize(
    "format, quality, lossless",
    [
        (
            tiledb.filter.lt.WebpInputFormat.WEBP_RGB,
            100.0,
            False,
        ),  # Test setting format with enum values
        (tiledb.filter.lt.WebpInputFormat.WEBP_BGR, 50.0, True),
        (tiledb.filter.lt.WebpInputFormat.WEBP_RGBA, 25.5, False),
        (4, 0.0, True),  # Test setting format with integral type
    ],
)
def test_webp_ctor(format, quality, lossless):
    webp_filter = tiledb.WebpFilter(
        input_format=format, quality=quality, lossless=lossless
    )
    np.testing.assert_equal(
        webp_filter.input_format, tiledb.filter.lt.WebpInputFormat(format)
    )
    np.testing.assert_equal(webp_filter.quality, quality)
    np.testing.assert_equal(webp_filter.lossless, lossless)


@pytest.mark.skipif(
    not main.test_webp_filter.webp_filter_exists(),
    reason="Can't create WebP filter; built with TILEDB_WEBP=OFF",
)
@pytest.mark.parametrize(
    "attr_dtype, dim_dtype, var, sparse",
    [
        (np.int64, np.int64, None, True),  # Sparse arrays are not supported
        (np.int64, np.int64, True, False),  # Variable attributes are not supported
    ],
)
def test_webp_init(attr_dtype, dim_dtype, var, sparse):
    with pytest.raises(tiledb.TileDBError):
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                [
                    tiledb.Dim("y", domain=(1, 100), dtype=dim_dtype),
                    tiledb.Dim("x", domain=(1, 300), dtype=dim_dtype),
                ]
            ),
            attrs=[
                tiledb.Attr(
                    "rgb", dtype=attr_dtype, var=var, filters=[tiledb.WebpFilter()]
                )
            ],
            sparse=sparse,
        )


def make_image_data(width, height, pixel_depth):
    center_x = width / 2
    center_y = height / 2

    colors = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "white": [255, 255, 255],
        "black": [0, 0, 0],
    }
    if pixel_depth > 3:
        for color in colors.values():
            color.append(255)

    rgb = []
    for row in range(0, height):
        r = []
        for col in range(0, width):
            if row < center_y and col < center_x:
                r.append(colors["red"])
            elif row < center_y and col > center_x:
                r.append(colors["green"])
            elif row > center_y and col < center_x:
                r.append(colors["blue"])
            elif row > center_y and col > center_x:
                r.append(colors["white"])
            elif row == center_y or col == center_x:
                r.append(colors["black"])
        rgb.append(r)
    return rgb


@pytest.mark.skipif(
    not main.test_webp_filter.webp_filter_exists(),
    reason="Can't create WebP filter; built with TILEDB_WEBP=OFF",
)
@pytest.mark.parametrize(
    "width, height",
    [
        (3, 7),
        (20, 20),
        (40, 40),
        (479, 149),
        (1213, 1357),
        (1111, 3333),
    ],
)
@pytest.mark.parametrize(
    "colorspace",
    [
        tiledb.filter.lt.WebpInputFormat.WEBP_RGB,
        tiledb.filter.lt.WebpInputFormat.WEBP_BGR,
        tiledb.filter.lt.WebpInputFormat.WEBP_RGBA,
        tiledb.filter.lt.WebpInputFormat.WEBP_BGRA,
    ],
)
@pytest.mark.parametrize("lossless", [True, False])
def test_webp_filter(width, height, colorspace, lossless):
    pixel_depth = (
        3 if int(colorspace) < int(tiledb.filter.lt.WebpInputFormat.WEBP_RGBA) else 4
    )
    data = make_image_data(width, height, pixel_depth)
    data = np.array(data, dtype=np.uint8).reshape(height, width * pixel_depth)

    y_tile = round(height / 2)
    x_tile = round(width / 2) * pixel_depth

    dim_dtype = np.min_scalar_type(data.size)
    dims = (
        tiledb.Dim(
            name="Y",
            domain=(1, height),
            dtype=dim_dtype,
            tile=y_tile,
        ),
        tiledb.Dim(
            name="X",
            domain=(1, width * pixel_depth),
            dtype=dim_dtype,
            tile=x_tile,
        ),
    )
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(
                name="rgb",
                dtype=np.uint8,
                filters=[
                    tiledb.WebpFilter(
                        input_format=colorspace, quality=100.0, lossless=lossless
                    )
                ],
            )
        ],
    )

    uri = tempfile.mkdtemp()
    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as A:
        A[:] = data
    with tiledb.open(uri, "r") as A:
        read_image = A[:]

    if lossless:
        assert_array_equal(
            data.reshape(np.array(read_image["rgb"]).shape), read_image["rgb"]
        )
    else:
        assert_allclose(
            data.reshape(np.array(read_image["rgb"]).shape), read_image["rgb"], 125
        )

    # Cleanup.
    shutil.rmtree(uri)
