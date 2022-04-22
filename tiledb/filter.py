import io
from typing import List, overload, Sequence, TYPE_CHECKING, Union

import tiledb.cc as lt
from .ctx import default_ctx

if TYPE_CHECKING:
    from .libtiledb import Ctx


class Filter(lt.Filter):
    """Base class for all TileDB filters."""

    def __init__(self, type: lt.FilterOption, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()

        super().__init__(lt.Context(self._ctx.__capsule__(), False), type)

    def __repr__(self) -> str:
        output = io.StringIO()
        output.write(f"{type(self).__name__}(")
        if hasattr(self, "_attrs_"):
            for f in self._attrs_():
                a = getattr(self, f)
                output.write(f"{f}={a}")
        output.write(")")
        return output.getvalue()

    def _repr_html_(self) -> str:
        output = io.StringIO()

        opt = list(self._attrs_().keys())[0] if self._attrs_() else ""
        val = getattr(self, opt) if self._attrs_() else ""

        output.write("<section>\n")
        output.write("<table>\n")
        output.write("<tr>\n")
        output.write(f"<td>{type(self).__name__}</td>\n")
        output.write(f"<td>{opt}</td>")
        output.write(f"<td>{val}</td>")
        output.write("</tr>\n")
        output.write("</table>\n")
        output.write("</section>\n")

        return output.getvalue()

    def __eq__(self, other: "Filter"):
        if other.__class__ is not self.__class__:
            return False
        for f in self._attrs_():
            left = getattr(self, f)
            right = getattr(other, f)
            if left != right:
                return False
        return True


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

    def __init__(self, type: lt.FilterType, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(type, self._ctx)
        self._set_option(
            lt.Context(self._ctx.__capsule__(), False),
            lt.FilterOption.COMPRESSION_LEVEL,
            self._level,
        )

    @property
    def level(self):
        return self._level


class NoOpFilter(Filter):
    """A filter that does nothing."""

    def __init__(self, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.NONE, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.GZIP, self._level, self._ctx)

    def _attrs_(self):
        return {"level": self.level}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.ZSTD, self._level, self._ctx)

    def _attrs_(self):
        return {"level": self.level}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.LZ4, self._level, self._ctx)

    def _attrs_(self):
        return {"level": self.level}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.BZIP2, self._level, self._ctx)

    def _attrs_(self):
        return {"level": self.level}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.RLE, self._level, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, level: int = -1, ctx: "Ctx" = None):
        self._level = level
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.DOUBLE_DELTA, self._level, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        super().__init__(lt.FilterType.BITSHUFFLE, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        super().__init__(lt.FilterType.BYTESHUFFLE, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, window: int = -1, ctx: "Ctx" = None):
        self._window = window
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.BIT_WIDTH_REDUCTION)
        self._set_option(
            lt.Context(self._ctx.__capsule__(), False),
            lt.FilterOption.BIT_WIDTH_MAX_WINDOW,
            self._window,
        )

    def _attrs_(self):
        return {"window": self._window}

    @property
    def window(self):
        return self._window


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

    def __init__(self, window: int = -1, ctx: "Ctx" = None):
        self._window = window
        self._ctx = ctx or default_ctx()

        super().__init__(lt.FilterType.POSITIVE_DELTA)
        self._set_option(
            lt.Context(self._ctx.__capsule__(), False),
            lt.FilterOption.POSITIVE_DELTA_MAX_WINDOW,
            self._window,
        )

    def _attrs_(self):
        return {"window": self._window}

    @property
    def window(self):
        return self._window


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

    def __init__(self, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        super().__init__(lt.FilterType.CHECKSUM_MD5, self._ctx)

    def _attrs_(self):
        return {}


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

    def __init__(self, ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        super().__init__(lt.FilterType.CHECKSUM_SHA256, self._ctx)

    def _attrs_(self):
        return {}


#
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

    def __init__(
        self,
        filters: Sequence[Filter] = None,
        chunksize: int = None,
        ctx: "Ctx" = None,
        is_capsule: bool = False,
    ):
        self._ctx = ctx or default_ctx()
        _cctx = lt.Context(self._ctx.__capsule__(), False)

        if is_capsule:
            super().__init__(_cctx, filters)
        else:
            super().__init__(_cctx)

            if filters is not None:
                filters = list(filters)
                for f in filters:
                    if not isinstance(f, Filter):
                        raise ValueError(
                            "filters argument must be an iterable of TileDB filter objects"
                        )
                    self._add_filter(f)

        if chunksize is not None:
            self._chunksize = chunksize

    @property
    def chunksize(self):
        return self._chunksize

    @overload
    def __getitem__(self, idx: int) -> Filter:
        ...

    @overload
    def __getitem__(self, idx: slice) -> List[Filter]:
        ...

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

    def __eq__(self, other: Union["FilterList", Sequence[Filter]]) -> bool:
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i, f in enumerate(self):
            if f != other[i]:
                return False
        return True

    def __len__(self) -> int:
        """
        :rtype: int
        :return: Number of filters in the FilterList

        """
        return self._nfilters()

    def append(self, filter: Filter):
        """
        :param Filter filter: the filter to append into the FilterList
        :raises ValueError: filter argument incorrect type

        """
        if not isinstance(filter, Filter):
            raise ValueError("filter argument must be a TileDB filter objects")
        self._add_filter(filter)

    def __repr__(self) -> str:
        filters = ",\n       ".join(
            [repr(self._getfilter(i)) for i in range(len(self))]
        )
        return "FilterList([{0!s}])".format(filters)

    def _repr_html_(self) -> str:
        if len(self) == 0:
            return "-"

        output = io.StringIO()
        output.write("<section>\n")
        output.write("<table>\n")
        output.write("<tr>\n")
        output.write("<th>Name</th>\n")
        output.write("<th>Option</th>\n")
        output.write("<th>Level</th>\n")
        output.write("</tr>\n")
        for filter in self:
            opt = list(filter._attrs_().keys())[0] if filter._attrs_() else "-"
            val = getattr(filter, opt) if filter._attrs_() else "-"
            output.write("<tr>\n")
            output.write(f"<td>{type(filter).__name__}</td>\n")
            output.write(f"<td>{opt}</td>")
            output.write(f"<td>{val}</td>")
            output.write("</tr>\n")
        output.write("</table>\n")
        output.write("</section>\n")

        return output.getvalue()

    def _getfilter(self, i: int) -> Filter:
        fil = self._filter(i)
        filtype = self.filter_type_cc_to_python[fil._type]
        opt = None

        if issubclass(filtype, CompressionFilter):
            opt = lt.FilterOption.COMPRESSION_LEVEL
        elif filtype == BitWidthReductionFilter:
            opt = lt.FilterOption.BIT_WIDTH_MAX_WINDOW
        elif filtype == PositiveDeltaFilter:
            opt = lt.FilterOption.POSITIVE_DELTA_MAX_WINDOW

        if opt is not None:
            _cctx = lt.Context(self._ctx.__capsule__(), False)
            filter = filtype(fil._get_option(_cctx, opt), ctx=self._ctx)
        else:
            filter = filtype(self._ctx)

        return filter
