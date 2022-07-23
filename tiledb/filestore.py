from typing import ByteString, TYPE_CHECKING
import warnings

import tiledb.cc as lt
from .ctx import default_ctx

if TYPE_CHECKING:
    from .libtiledb import Ctx


class Filestore:
    """
    Functions to set and get data to and from a TileDB Filestore Array.

    :param str uri: The URI to the TileDB Fileshare Array
    :param tiledb.Ctx ctx: A TileDB context
    """

    def __init__(self, uri: str, ctx: "Ctx" = None) -> None:
        self._ctx = ctx or default_ctx()
        self._filestore_uri = uri

    def write(self, buffer: ByteString, mime_type: str = "AUTODETECT") -> None:
        """
        Import data from an object that supports the buffer protocol to a Filestore Array.

        :param buffer ByteString: Data of type bytes, bytearray, memoryview, etc.
        :param str mime_type: MIME types are "AUTODETECT" (default), "image/tiff", "application/pdf"

        """
        try:
            buffer = memoryview(buffer)
        except TypeError:
            raise TypeError(
                f"Unexpected buffer type: buffer must support buffer protocol"
            )

        if not isinstance(mime_type, str):
            raise TypeError(
                f"Unexpected mime_type type '{type(mime_type)}': expected str"
            )

        try:
            lt.Filestore._buffer_import(
                lt.Context(self._ctx, False),
                self._filestore_uri,
                buffer,
                lt.Filestore._mime_type_from_str(mime_type),
            )
        except Exception as e:
            raise (e)

    def read(self, offset: int = 0, size: int = -1) -> bytes:
        """
        :param int offset: Byte position to begin reading. Defaults to beginning of filestore.
        :param int size: Total number of bytes to read. Defaults to -1 which reads the entire filestore.
        :rtype: bytes
        :return: Data from the Filestore Array

        """
        if not isinstance(offset, int):
            raise TypeError(f"Unexpected offset type '{type(offset)}': expected int")

        if not isinstance(size, int):
            raise TypeError(f"Unexpected size type '{type(size)}': expected int")

        if size == -1:
            size = len(self)
        size = max(size - offset, 0)

        return lt.Filestore._buffer_export(
            lt.Context(self._ctx, False),
            self._filestore_uri,
            offset,
            size,
        )

    @staticmethod
    def copy_from(
        filestore_array_uri: str,
        file_uri: str,
        mime_type: str = "AUTODETECT",
        ctx: "Ctx" = None,
    ) -> None:
        """
        Copy data from a file to a Filestore Array.

        :param str filestore_array_uri: The URI to the TileDB Fileshare Array
        :param str file_uri: URI of file to export
        :param str mime_type: MIME types are "AUTODETECT" (default), "image/tiff", "application/pdf"
        :param tiledb.Ctx ctx: A TileDB context

        """
        if not isinstance(filestore_array_uri, str):
            raise TypeError(
                f"Unexpected filestore_array_uri type '{type(filestore_array_uri)}': expected str"
            )

        if not isinstance(file_uri, str):
            raise TypeError(
                f"Unexpected file_uri type '{type(file_uri)}': expected str"
            )

        if not isinstance(mime_type, str):
            raise TypeError(
                f"Unexpected mime_type type '{type(mime_type)}': expected str"
            )

        ctx = ctx or default_ctx()

        lt.Filestore._uri_import(
            lt.Context(ctx, False),
            filestore_array_uri,
            file_uri,
            lt.Filestore._mime_type_from_str(mime_type),
        )

    @staticmethod
    def copy_to(filestore_array_uri: str, file_uri: str, ctx: "Ctx" = None) -> None:
        """
        Copy data from a Filestore Array to a file.

        :param str filestore_array_uri: The URI to the TileDB Fileshare Array
        :param str file_uri: The URI to the TileDB Fileshare Array
        :param tiledb.Ctx ctx: A TileDB context

        """
        if not isinstance(filestore_array_uri, str):
            raise TypeError(
                f"Unexpected filestore_array_uri type '{type(filestore_array_uri)}': expected str"
            )

        if not isinstance(file_uri, str):
            raise TypeError(
                f"Unexpected file_uri type '{type(file_uri)}': expected str"
            )

        ctx = ctx or default_ctx()

        lt.Filestore._uri_export(lt.Context(ctx, False), filestore_array_uri, file_uri)

    def uri_import(self, file_uri: str, mime_type: str = "AUTODETECT") -> None:
        warnings.warn(
            "Filestore.uri_import is deprecated; please use Filestore.copy_from",
            DeprecationWarning,
        )

        if not isinstance(file_uri, str):
            raise TypeError(
                f"Unexpected file_uri type '{type(file_uri)}': expected str"
            )

        if not isinstance(mime_type, str):
            raise TypeError(
                f"Unexpected mime_type type '{type(mime_type)}': expected str"
            )

        lt.Filestore._uri_import(
            lt.Context(self._ctx, False),
            self._filestore_uri,
            file_uri,
            lt.Filestore._mime_type_from_str(mime_type),
        )

    def __len__(self) -> int:
        """
        :rtype: int
        :return: Bytes in the Filestore Array

        """
        return lt.Filestore._size(lt.Context(self._ctx, False), self._filestore_uri)
