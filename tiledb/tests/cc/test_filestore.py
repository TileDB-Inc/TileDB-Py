import os

import pytest

import tiledb
import tiledb.cc as lt


@pytest.fixture
def text_fname(tmp_path):
    path = os.path.join(tmp_path, "text_fname")
    vfs = tiledb.VFS()
    vfs.touch(path)
    with vfs.open(path, "wb") as fio:
        fio.write(b"Simple text file.\n")
        fio.write(b"With two lines.")
    return path


def test_lt_schema_create(text_fname):
    ctx = lt.Context()
    schema = lt.Filestore._schema_create(ctx, text_fname)
    assert type(schema) == lt.ArraySchema


def test_libtiledb_schema_create_buffer(tmp_path, text_fname):
    ctx = lt.Context()
    path = os.path.join(tmp_path, "test_libtiledb_schema_create_buffer")
    schema = tiledb.ArraySchema.from_file(text_fname)
    tiledb.Array.create(path, schema)

    data = b"buffer"
    lt.Filestore._buffer_import(ctx, path, data, lt.MIMEType.AUTODETECT)
    assert bytes(data) == lt.Filestore._buffer_export(ctx, path, 0, len(data))
    assert lt.Filestore._size(ctx, path) == len(data)

    output_file = os.path.join(tmp_path, "output_file")
    vfs = tiledb.VFS()
    vfs.touch(output_file)
    lt.Filestore._uri_export(ctx, path, output_file)
    with vfs.open(output_file, "rb") as fio:
        assert fio.read() == data


def test_libtiledb_schema_create_uri(tmp_path, text_fname):
    ctx = lt.Context()
    path = os.path.join(tmp_path, "test_libtiledb_schema_create_uri")
    schema = tiledb.ArraySchema.from_file(text_fname)
    tiledb.Array.create(path, schema)

    lt.Filestore._uri_import(ctx, path, text_fname, lt.MIMEType.AUTODETECT)
    with open(text_fname, "rb") as text:
        data = text.read()
        assert data == lt.Filestore._buffer_export(ctx, path, 0, len(data))
        assert lt.Filestore._size(ctx, path) == len(data)


def test_mime_type():
    to_str = {
        lt.MIMEType.AUTODETECT: "AUTODETECT",
        lt.MIMEType.TIFF: "image/tiff",
        lt.MIMEType.PDF: "application/pdf",
    }

    for k in to_str:
        assert lt.Filestore._mime_type_to_str(k) == to_str[k]

    from_str = {
        "AUTODETECT": lt.MIMEType.AUTODETECT,
        "image/tiff": lt.MIMEType.TIFF,
        "application/pdf": lt.MIMEType.PDF,
    }

    for k in from_str:
        assert lt.Filestore._mime_type_from_str(k) == from_str[k]
