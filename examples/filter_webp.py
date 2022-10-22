import os
import shutil
import tiledb
import tifffile
import numpy as np
from numpy.testing import assert_array_equal


def write_image(uri: str, image: np.ndarray) -> None:
    if os.path.exists(uri):
        shutil.rmtree(uri)

    dim_dtype = np.min_scalar_type(image.size)
    print("Writing image")

    height = image.shape[0]
    y_bound = height
    y_tile = min(height, 1024)

    width = image.shape[1]
    x_bound = width
    x_bound *= 3
    x_tile = min(width, 1024) * 3

    # Print some debug info
    print(f"height: {height}; width: {width}")
    print(f"y_bound: {y_bound}; x_bound: {x_bound}; bytes: {y_bound * x_bound}")
    print(f"y_tile: {y_tile}; x_tile: {x_tile}")
    print(f"Initial image shape: {image.shape}")
    # if image.ndim > 2:
    #     image = image.reshape((height, width * 3))
    #     print(f"Updated image shape: {image.shape}")

    dims = (tiledb.Dim(name='Y',
                       domain=(1, y_bound),
                       dtype=dim_dtype,
                       tile=y_tile,),
            tiledb.Dim(name='X',
                       domain=(1, x_bound),
                       dtype=dim_dtype,
                       tile=x_tile,))

    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(
                name="rgb",
                dtype=image.dtype,
                # filters=[tiledb.ZstdFilter(level=9)],
                filters=[tiledb.WebpFilter(input_format=1, quality=100.0, lossless=lossless)],
                # filters=[tiledb.WebpFilter(input_format=1, quality=100.0, lossless=lossless), tiledb.ZstdFilter(level=9)],
            )
        ],
    )

    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as A:
        print("Start writing")
        A[:] = image
    print("Wrote image")
    print(image)

    with tiledb.open(uri, "r") as A:
        print("Start reading")
        read_image = A[:]
    print("Read image")

    if lossless:
        assert_array_equal(image.reshape(np.array(read_image['rgb']).shape), read_image['rgb'])
        print("Lossless assertion passed")
    else:
        print("TODO")


def convert_image(tiff_sr, output_path: str, level_min: int = 0):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    tiledb.group_create(output_path)

    # Create a TileDB array for each level in range(level_min, reader.level_count)
    uris = []
    for level in range(level_min, len(tiff_sr)):
        image = tiff_sr[level].asarray()
        uri = os.path.join(output_path, f"l_{level}.tdb")
        write_image(uri, image)
        uris.append(uri)


lossless = True
file_name = "/home/shaun/Downloads/C3L-02660-26.svs"
output_path = "/home/shaun/Pictures/webp-filter/"

print(f"opening tiff: {file_name}")
# tiff_series = tifffile.imread(file_name, series=0, level=0)
tiff_series = tifffile.TiffFile(file_name).series
print("opened tiff file")
img = tiff_series[0].asarray()
print(f"Loaded image with size: {img.size}; Press enter to continue")
input()

print("Converting image to TileDB Array")
# Level 0 with this image segfaults
convert_image(tiff_series, output_path, 0)
# write_image(output_path, img)
# write_image(output_path, tiff_series)
print("Image converted to TileDB Array")
