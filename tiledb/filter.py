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

from typing import Sequence, Union


def filter_list_eq(self, other: Union["FilterList", Sequence[Filter]]) -> bool:
    if other is None:
        return False
    if len(self) != len(other):
        return False
    for i, f in enumerate(self):
        if f != other[i]:
            return False
    return True


FilterList.__eq__ = filter_list_eq
