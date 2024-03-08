import io
import os
from types import TracebackType
from typing import List, Optional, Type, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Config, Ctx, default_ctx

_AnyPath = Union[str, bytes, os.PathLike]


class VFS(lt.VFS):
    """TileDB VFS class

    Encapsulates the TileDB VFS module instance with a specific configuration (config).

    :param tiledb.Ctx ctx: The TileDB Context
    :param config: Override `ctx` VFS configurations with updated values in config.
    :type config: tiledb.Config or dict

    """

    def __init__(self, config: Union[Config, dict] = None, ctx: Optional[Ctx] = None):
        ctx = ctx or default_ctx()

        if config:
            from .libtiledb import Config

            if isinstance(config, Config):
                config = config.dict()
            else:
                try:
                    config = dict(config)
                except Exception:
                    raise ValueError("`config` argument must be of type Config or dict")

            ccfg = lt.Config(config)
            super().__init__(ctx, ccfg)
        else:
            super().__init__(ctx)

    def ctx(self) -> Ctx:
        """
        :rtype: tiledb.Ctx
        :return: context associated with the VFS object
        """
        return self._ctx

    def config(self) -> Config:
        """
        :rtype: tiledb.Config
        :return: config associated with the VFS object
        """
        return self._config

    def open(self, uri: _AnyPath, mode: str = "rb"):
        """Opens a VFS file resource for reading / writing / appends at URI.

        If the file did not exist upon opening, a new file is created.

        :param str uri: URI of VFS file resource
        :param mode str: 'rb' for opening the file to read, 'wb' to write, 'ab' to append
        :rtype: FileHandle
        :return: TileDB FileIO
        :raises TypeError: cannot convert `uri` to unicode string
        :raises ValueError: invalid mode
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return FileIO(self, uri, mode)

    def close(self, file: lt.FileHandle):
        """Closes a VFS FileHandle object.

        :param FileIO file: An opened VFS FileIO
        :rtype: FileIO
        :return: closed VFS FileHandle
        :raises: :py:exc:`tiledb.TileDBError`

        """
        file.close()
        return file

    def write(self, file: lt.FileHandle, buff: Union[str, bytes]):
        """Writes buffer to opened VFS FileHandle.

        :param FileHandle file: An opened VFS FileHandle in 'w' mode
        :param buff: a Python object that supports the byte buffer protocol
        :raises TypeError: cannot convert buff to bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if isinstance(file, FileIO):
            raise lt.TileDBError(
                "`tiledb.VFS().open` now returns a a FileIO object. Use "
                "`FileIO.write`. This message will be removed in 0.21.0.",
            )
        if isinstance(buff, str):
            buff = buff.encode()
        file.write(buff)

    def read(self, file: lt.FileHandle, offset: int, nbytes: int) -> bytes:
        """Read nbytes from an opened VFS FileHandle at a given offset.

        :param FileHandle file: An opened VFS FileHandle in 'r' mode
        :param int offset: offset position in bytes to read from
        :param int nbytes: number of bytes to read
        :rtype: :py:func:`bytes`
        :return: read bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if isinstance(file, FileIO):
            raise lt.TileDBError(
                "`tiledb.VFS().open` now returns a a FileIO object. Use "
                "`FileIO.seek` and `FileIO.read`. This message will be removed "
                "in 0.21.0."
            )

        if nbytes == 0:
            return b""

        return file.read(offset, nbytes)

    def supports(self, scheme: str) -> bool:
        """Returns true if the given URI scheme (storage backend) is supported.

        :param str scheme: scheme component of a VFS resource URI (ex. 'file' / 'hdfs' / 's3')
        :rtype: bool
        :return: True if the linked libtiledb version supports the storage backend, False otherwise
        :raises ValueError: VFS storage backend is not supported

        """
        if scheme == "file":
            return True

        scheme_to_fs_type = {
            "s3": lt.FileSystem.S3,
            "azure": lt.FileSystem.AZURE,
            "gcs": lt.FileSystem.GCS,
            "hdfs": lt.FileSystem.HDFS,
        }

        if scheme not in scheme_to_fs_type:
            raise ValueError(f"Unsupported VFS scheme '{scheme}://'")

        return self._ctx.is_supported_fs(scheme_to_fs_type[scheme])

    def create_bucket(self, uri: _AnyPath):
        """Creates an object store bucket with the input URI.

        :param str uri: Input URI of the bucket

        """
        return self._create_bucket(_to_path_str(uri))

    def remove_bucket(self, uri: _AnyPath):
        """Deletes an object store bucket with the input URI.

        :param str uri: Input URI of the bucket

        """
        return self._remove_bucket(_to_path_str(uri))

    def is_bucket(self, uri: _AnyPath) -> bool:
        """
        :param str uri: Input URI of the bucket
        :rtype: bool
        :return: True if an object store bucket with the input URI exists, False otherwise

        """
        return self._is_bucket(_to_path_str(uri))

    def empty_bucket(self, uri: _AnyPath):
        """Empty an object store bucket.

        :param str uri: Input URI of the bucket

        """
        return self._empty_bucket(_to_path_str(uri))

    def is_empty_bucket(self, uri: _AnyPath) -> bool:
        """
        :param str uri: Input URI of the bucket
        :rtype: bool
        :return: True if an object store bucket is empty, False otherwise

        """
        return self._is_empty_bucket(_to_path_str(uri))

    def create_dir(self, uri: _AnyPath):
        """Check if an object store bucket is empty.

        :param str uri: Input URI of the bucket

        """
        return self._create_dir(_to_path_str(uri))

    def is_dir(self, uri: _AnyPath) -> bool:
        """
        :param str uri: Input URI of the directory
        :rtype: bool
        :return: True if a directory with the input URI exists, False otherwise

        """
        return self._is_dir(_to_path_str(uri))

    def remove_dir(self, uri: _AnyPath):
        """Removes a directory (recursively) with the input URI.

        :param str uri: Input URI of the directory

        """
        return self._remove_dir(_to_path_str(uri))

    def dir_size(self, uri: _AnyPath) -> int:
        """
        :param str uri: Input URI of the directory
        :rtype: int
        :return: The size of a directory with the input URI

        """
        return self._dir_size(_to_path_str(uri))

    def move_dir(self, old_uri: _AnyPath, new_uri: _AnyPath):
        """Renames a TileDB directory from an old URI to a new URI.

        :param str old_uri: Input of the old directory URI
        :param str new_uri: Input of the new directory URI

        """
        return self._move_dir(_to_path_str(old_uri), _to_path_str(new_uri))

    def copy_dir(self, old_uri: _AnyPath, new_uri: _AnyPath):
        """Copies a TileDB directory from an old URI to a new URI.

        :param str old_uri: Input of the old directory URI
        :param str new_uri: Input of the new directory URI

        """
        return self._copy_dir(_to_path_str(old_uri), _to_path_str(new_uri))

    def is_file(self, uri: _AnyPath) -> bool:
        """
        :param str uri: Input URI of the file
        :rtype: bool
        :return: True if a file with the input URI exists, False otherwise

        """
        return self._is_file(_to_path_str(uri))

    def remove_file(self, uri: _AnyPath):
        """Removes a file with the input URI.

        :param str uri: Input URI of the file

        """
        return self._remove_file(_to_path_str(uri))

    def file_size(self, uri: _AnyPath) -> int:
        """
        :param str uri: Input URI of the file
        :rtype: int
        :return: The size of a file with the input URI

        """
        return self._file_size(_to_path_str(uri))

    def move_file(self, old_uri: _AnyPath, new_uri: _AnyPath):
        """Renames a TileDB file from an old URI to a new URI.

        :param str old_uri: Input of the old file URI
        :param str new_uri: Input of the new file URI

        """
        return self._move_file(_to_path_str(old_uri), _to_path_str(new_uri))

    def copy_file(self, old_uri: _AnyPath, new_uri: _AnyPath):
        """Copies a TileDB file from an old URI to a new URI.

        :param str old_uri: Input of the old file URI
        :param str new_uri: Input of the new file URI

        """
        return self._copy_file(_to_path_str(old_uri), _to_path_str(new_uri))

    def ls(self, uri: _AnyPath) -> List[str]:
        """Retrieves the children in directory `uri`. This function is
        non-recursive, i.e., it focuses in one level below `uri`.

        :param str uri: Input URI of the directory
        :rtype: List[str]
        :return: The children in directory `uri`

        """
        return self._ls(_to_path_str(uri))

    def touch(self, uri: _AnyPath):
        """Touches a file with the input URI, i.e., creates a new empty file.

        :param str uri: Input URI of the file

        """
        return self._touch(_to_path_str(uri))


class FileIO(io.RawIOBase):
    """TileDB FileIO class that encapsulates files opened by tiledb.VFS. The file
    operations are meant to mimic Python's built-in file I/O methods."""

    def __init__(self, vfs: VFS, uri: _AnyPath, mode: str = "rb"):
        uri = _to_path_str(uri)
        self._vfs = vfs

        str_to_vfs_mode = {
            "rb": lt.VFSMode.READ,
            "wb": lt.VFSMode.WRITE,
            "ab": lt.VFSMode.APPEND,
        }
        if mode not in str_to_vfs_mode:
            raise ValueError(f"invalid mode {mode}")

        self._mode = mode
        self._offset = 0
        self._nbytes = 0

        if self._mode == "rb":
            try:
                self._nbytes = vfs.file_size(uri)
            except Exception as e:
                raise lt.TileDBError(f"URI {uri!r} is not a valid file") from e

        self._fh = lt.FileHandle(
            self._vfs._ctx, self._vfs, uri, str_to_vfs_mode[self._mode]
        )

    def __len__(self):
        """
        :rtype: int
        :return: Number of bytes in file

        """
        return self._nbytes

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self._fh._close()

    @property
    def mode(self) -> str:
        """
        :rtype: str
        :return: Whether the file is in read mode ("rb"), write mode ("wb"), or append mode ("ab")

        """
        return self._mode

    def readable(self) -> bool:
        """
        :rtype: bool
        :return: True if the file is readable (ie. "rb" mode), otherwise False

        """
        return self._mode == "rb"

    def writable(self) -> bool:
        """
        :rtype: bool
        :return: True if the file is writable (ie. "wb" or "ab" mode), otherwise False

        """
        return self._mode != "rb"

    @property
    def closed(self) -> bool:
        """
        :rtype: bool
        :return: True if the file is closed, otherwise False

        """
        return self._fh._closed

    def seekable(self):
        """
        All tiledb.FileIO objects are seekable.

        :rtype: bool
        :return: True

        """
        return True

    def flush(self):
        """
        Force the data to be written to the file.
        """
        self._fh._flush()

    def seek(self, offset: int, whence: int = 0):
        """
        :param int offset: Byte position to set the file pointer
        :param int whence: Reference point. A whence value of 0 measures from the
        beginning of the file, 1 uses the current file position, and 2 uses the
        end of the file as the reference point. whence can be omitted and defaults to 0.
        """
        if not isinstance(offset, int):
            raise TypeError(
                f"Offset must be an integer or None (got type {type(offset)})"
            )
        if whence == 0:
            if offset < 0:
                raise ValueError(
                    "offset must be a positive or zero value when SEEK_SET"
                )
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = self._nbytes + offset
        else:
            raise ValueError("whence must be equal to SEEK_SET, SEEK_START, SEEK_END")
        if self._offset < 0:
            self._offset = 0
        elif self._offset > self._nbytes:
            self._offset = self._nbytes

        return self._offset

    def tell(self) -> int:
        """
        :rtype: int
        :return: The current position in the file represented as number of bytes

        """
        return self._offset

    def read(self, size: int = -1) -> bytes:
        """
        Read the file from the current pointer position.

        :param int size: Number of bytes to read. By default, size is set to -1
        which will read until the end of the file.
        :rtype: bytes
        :return: The bytes in the file

        """
        if not isinstance(size, int):
            raise TypeError(f"size must be an integer or None (got type {type(size)})")
        if not self.readable():
            raise IOError("Cannot read from write-only FileIO handle")
        if self.closed:
            raise IOError("Cannot read from closed FileIO handle")

        nbytes_left = self._nbytes - self._offset
        nbytes = nbytes_left if size < 0 or size > nbytes_left else size
        if nbytes == 0:
            return b""

        buff = self._fh._read(self._offset, nbytes)
        self._offset += nbytes
        return buff

    def write(self, buff: bytes):
        """
        :param bytes buff: Write the given bytes to the file

        """
        if not self.writable():
            raise IOError("cannot write to read-only FileIO handle")
        if isinstance(buff, str):
            buff = buff.encode()
        nbytes = len(buff)
        self._fh._write(buff)
        self._nbytes += nbytes
        self._offset += nbytes
        return nbytes

    def readinto(self, buff: np.ndarray) -> int:
        """
        Read bytes into a pre-allocated, writable bytes-like object b, and return the number of bytes read.

        :param buff bytes:
        """
        buff = memoryview(buff).cast("b")
        size = len(buff)
        if not self.readable():
            raise IOError("Cannot read from write-only FileIO handle")
        if self.closed:
            raise IOError("Cannot read from closed FileIO handle")

        nbytes_left = self._nbytes - self._offset
        nbytes = nbytes_left if size > nbytes_left else size
        if nbytes == 0:
            return None

        buff_temp = self._fh._read(self._offset, nbytes)
        self._offset += nbytes
        buff[: len(buff_temp)] = memoryview(buff_temp).cast("b")
        return len(buff_temp)

    def readinto1(self, b):
        return self.readinto(b)


def _to_path_str(pth: _AnyPath) -> Union[str, bytes]:
    if isinstance(pth, (str, bytes)):
        return pth
    try:
        return pth.__fspath__()
    except AttributeError as ae:
        raise TypeError(
            "VFS paths must be strings, bytes, or os.PathLike objects"
        ) from ae
