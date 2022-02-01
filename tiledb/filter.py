from dataclasses import dataclass, field
import io
from typing import Sequence

import tiledb.cc as lt
from .ctx import default_ctx
from .libtiledb import Ctx


@dataclass(repr=False)
class Filter(lt.Filter):
    """Base class for all TileDB filters."""

    type: lt.FilterType = lt.FilterType.NONE
    ctx: Ctx = field(default_factory=default_ctx, repr=False)

    def __post_init__(self, type=None):
        if type:
            self.type = type

        super().__init__(lt.Context(self.ctx.__capsule__(), False), self.type)

    def __repr__(self):
        output = io.StringIO()
        output.write(f"{type(self).__name__}(")
        if hasattr(self, "_attrs_"):
            for f in self._attrs_():
                a = getattr(self, f)
                output.write(f"{f}={a}")
        output.write(")")
        return output.getvalue()

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<section>\n")
        output.write("<table>\n")

        output.write("<tr>\n")
        output.write("<th></th>\n")
        if hasattr(self, "_attrs_"):
            for f in self._attrs_():
                output.write(f"<th>{f}</th>")
        output.write("</tr>\n")

        output.write("<tr>\n")
        output.write(f"<td>{type(self).__name__}</td>\n")
        if hasattr(self, "_attrs_"):
            for f in self._attrs_():
                output.write(f"<td>{getattr(self, f)}</td>")
        output.write("</tr>\n")

        output.write("</table>\n")
        output.write("</section>\n")

        return output.getvalue()


@dataclass(repr=False)
class CompressionFilter(Filter):
    """
    Base class for filters performing compression.

    All compression filters support a compression level option, although
    some (such as RLE) ignore it.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.GzipFilter(level=10)]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    type: lt.FilterType
    level: int = -1

    def __post_init__(self, type, level):
        if type:
            self.type = type

        if level:
            self.level = level

        super().__post_init__(self.type)
        self.set_option(
            lt.Context(self.ctx.__capsule__(), False),
            lt.FilterOption.COMPRESSION_LEVEL,
            self.level,
        )


@dataclass(repr=False)
class NoOpFilter(Filter):
    """A filter that does nothing."""

    def __post_init__(self):
        super().__post_init__(lt.FilterType.NONE)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class GzipFilter(CompressionFilter):
    """
    Filter that compresses using gzip.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.GzipFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.GZIP, self.level)

    def _attrs_(self):
        return {"level": self.level}


@dataclass(repr=False)
class ZstdFilter(CompressionFilter):
    """
    Filter that compresses using zstd.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ZstdFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.ZSTD, self.level)

    def _attrs_(self):
        return {"level": self.level}


@dataclass(repr=False)
class LZ4Filter(CompressionFilter):
    """
    Filter that compresses using lz4.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.LZ4Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.LZ4, self.level)

    def _attrs_(self):
        return {"level": self.level}


@dataclass(repr=False)
class Bzip2Filter(CompressionFilter):
    """
    Filter that compresses using bzip2.

    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.Bzip2Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.BZIP2, self.level)

    def _attrs_(self):
        return {"level": self.level}


@dataclass(repr=False)
class RleFilter(CompressionFilter):
    """
    Filter that compresses using run-length encoding (RLE).

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.RleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.RLE, self.level)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class DoubleDeltaFilter(CompressionFilter):
    """
    Filter that performs double-delta encoding.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.DoubleDeltaFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.DOUBLE_DELTA, self.level)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class BitShuffleFilter(Filter):
    """
    Filter that performs a bit shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.BitShuffleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.BITSHUFFLE)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class ByteShuffleFilter(Filter):
    """
    Filter that performs a byte shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ByteShuffleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.BYTESHUFFLE)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class BitWidthReductionFilter(Filter):
    """Filter that performs bit-width reduction.

     :param ctx: A TileDB Context
     :type ctx: tiledb.Ctx
     :param window: (default None) max window size for the filter
     :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.BitWidthReductionFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    window: int = -1

    def __post_init__(self):
        super().__post_init__(lt.FilterType.BIT_WIDTH_REDUCTION)
        self.set_option(
            lt.Context(self.ctx.__capsule__(), False),
            lt.FilterOption.BIT_WIDTH_MAX_WINDOW,
            self.window,
        )

    def _attrs_(self):
        return {"window": self.window}


@dataclass(repr=False)
class PositiveDeltaFilter(Filter):
    """
    Filter that performs positive-delta encoding.

    :param ctx: A TileDB Context
    :type ctx: tiledb.Ctx
    :param window: (default None) the max window for the filter
    :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.PositiveDeltaFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    window: int = -1

    def __post_init__(self):
        super().__post_init__(lt.FilterType.POSITIVE_DELTA)
        self.set_option(
            lt.Context(self.ctx.__capsule__(), False),
            lt.FilterOption.POSITIVE_DELTA_MAX_WINDOW,
            self.window,
        )

    def _attrs_(self):
        return {"window": self.window}


@dataclass(repr=False)
class ChecksumMD5Filter(Filter):
    """
    MD5 checksum filter.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ChecksumMD5Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.CHECKSUM_MD5)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class ChecksumSHA256Filter(Filter):
    """
    SHA256 checksum filter.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ChecksumSHA256Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __post_init__(self):
        super().__post_init__(lt.FilterType.CHECKSUM_SHA256)

    def _attrs_(self):
        return {}


@dataclass(repr=False)
class FilterList(lt.FilterList):
    """
    An ordered list of Filter objects for filtering TileDB data.

    FilterLists contain zero or more Filters, used for filtering attribute data, the array coordinate data, etc.

    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx
    :param filters: An iterable of Filter objects to add.
    :param chunksize: (default None) chunk size used by the filter list in bytes
    :type chunksize: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     # Create several filters
    ...     gzip_filter = tiledb.GzipFilter()
    ...     bw_filter = tiledb.BitWidthReductionFilter()
    ...     # Create a filter list that will first perform bit width reduction, then gzip compression.
    ...     filters = tiledb.FilterList([bw_filter, gzip_filter])
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64, filters=filters)
    ...     # Create a second attribute filtered only by gzip compression.
    ...     a2 = tiledb.Attr(name="a2", dtype=np.int64,
    ...                      filters=tiledb.FilterList([gzip_filter]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1, a2))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    filters: Sequence[Filter] = None
    chunksize: int = None
    ctx: Ctx = field(default_factory=default_ctx, repr=False)
    is_capsule: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.is_capsule:
            super().__init__(lt.Context(self.ctx.__capsule__(), False), self.filters)
            self.filters = [self._getfilter(i) for i in range(len(self))]
        else:
            super().__init__(lt.Context(self.ctx.__capsule__(), False))

            if self.filters is not None:
                self.filters = list(self.filters)
                for filter in self.filters:
                    if not isinstance(filter, Filter):
                        raise ValueError(
                            "filters argument must be an iterable of TileDB filter objects"
                        )
                    self.add_filter(filter)

        if self.chunksize:
            self._max_chunk_size = self.chunksize
        self.chunksize = self._max_chunk_size

    def __getitem__(self, idx):
        """Gets a copy of the filter in the list at the given index

        :param idx: index into the
        :type idx: int or slice
        :returns: A filter at given index / slice
        :raises IndexError: invalid index
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not isinstance(idx, (int, slice)):
            raise TypeError(
                "FilterList indices must be integers or slices, not {:s}".format(
                    type(idx).__name__
                )
            )

        if isinstance(idx, int):
            if idx < 0 or idx > (len(self) - 1):
                raise IndexError("FilterList index out of range")
            idx = slice(idx, idx + 1)
        else:
            if (
                not isinstance(idx.start, int)
                or not isinstance(idx.stop, int)
                or not isinstance(idx.step, int)
            ):
                raise IndexError("FilterList slice indices must be integers or None")

        filters = []
        (start, stop, step) = idx.indices(len(self))
        for i in range(start, stop, step):
            filters.append(self._getfilter(i))

        if len(filters) == 1:
            return filters[0]

        return filters

    def __eq__(self, other):
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i, f in enumerate(self):
            if f != other[i]:
                return False
        return True

    def append(self, filter):
        if not isinstance(filter, Filter):
            raise ValueError("filter argument must be a TileDB filter objects")
        self.add_filter(filter)

    def __repr__(self):
        filters = ",\n       ".join(
            [repr(self._getfilter(i)) for i in range(len(self))]
        )
        return "FilterList([{0!s}])".format(filters)

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<section>\n")
        for i in range(len(self)):
            output.write(self[i]._repr_html_())
        output.write("</section>\n")

        return output.getvalue()

    def _getfilter(self, i):
        filter_type_cc_to_python = {
            lt.FilterType.GZIP: GzipFilter,
            lt.FilterType.ZSTD: ZstdFilter,
            lt.FilterType.LZ4: LZ4Filter,
            lt.FilterType.BZIP2: Bzip2Filter,
            lt.FilterType.RLE: RleFilter,
            lt.FilterType.DOUBLE_DELTA: DoubleDeltaFilter,
            lt.FilterType.BIT_WIDTH_REDUCTION: BitWidthReductionFilter,
            lt.FilterType.BITSHUFFLE: BitShuffleFilter,
            lt.FilterType.BYTESHUFFLE: ByteShuffleFilter,
            lt.FilterType.POSITIVE_DELTA: PositiveDeltaFilter,
            lt.FilterType.CHECKSUM_MD5: ChecksumMD5Filter,
            lt.FilterType.CHECKSUM_SHA256: ChecksumSHA256Filter,
            lt.FilterType.NONE: NoOpFilter,
        }

        filtype = filter_type_cc_to_python[self.filter(i).type]
        if issubclass(filtype, CompressionFilter):
            level = self.filter(i).get_option(
                lt.Context(self.ctx.__capsule__(), False),
                lt.FilterOption.COMPRESSION_LEVEL,
            )
            fil = filtype(level=level, ctx=self.ctx)
        elif filtype == BitWidthReductionFilter:
            window = self.filter(i).get_option(
                lt.Context(self.ctx.__capsule__(), False),
                lt.FilterOption.BIT_WIDTH_MAX_WINDOW,
            )
            fil = filtype(window=window, ctx=self.ctx)
        elif filtype == PositiveDeltaFilter:
            window = self.filter(i).get_option(
                lt.Context(self.ctx.__capsule__(), False),
                lt.FilterOption.POSITIVE_DELTA_MAX_WINDOW,
            )
            fil = filtype(window=window, ctx=self.ctx)
        else:
            fil = filtype(self.ctx)

        return fil
