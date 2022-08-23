from .cc import (
    Filter,
    FilterList,
    NoOpFilter,
    GzipFilter,
    ZstdFilter,
    LZ4Filter,
    Bzip2Filter,
    RleFilter,
    DoubleDeltaFilter,
    DictionaryFilter,
    BitShuffleFilter,
    ByteShuffleFilter,
    BitWidthReductionFilter,
    PositiveDeltaFilter,
    ChecksumMD5Filter,
    ChecksumSHA256Filter,
)

from typing import Sequence, TYPE_CHECKING

import tiledb.cc as lt
from .ctx import default_ctx

if TYPE_CHECKING:
    from .libtiledb import Ctx


def filter_list_init(
    filters: Sequence[Filter] = None,
    chunksize: int = None,
    ctx: "Ctx" = None,
    is_capsule=False,
):
    ctx = ctx or default_ctx()
    cctx = lt.Context(ctx.__capsule__(), False)

    if is_capsule:
        FilterList(cctx, filters)
    else:
        FilterList(filters, chunksize, cctx)


# FilterList.__init__ = filter_list_init
