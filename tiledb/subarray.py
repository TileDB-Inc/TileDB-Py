from typing import (
    TYPE_CHECKING,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    List,
    Any,
    Union,
    cast,
)
import numpy as np
from numbers import Real

import tiledb.cc as lt
from .libtiledb import Array
from .ctx import CtxMixin
from ._array import ArrayImpl

if TYPE_CHECKING:
    from libtiledb import Ctx

# sentinel value to denote selecting an empty range
EmptyRange = object()

# TODO: expand with more accepted scalar types
Scalar = Real
Range = Tuple[Scalar, Scalar]


def to_scalar(obj: Any) -> Scalar:
    if np.isscalar(obj):
        return cast(Scalar, obj)
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return cast(Scalar, obj[()])
    raise ValueError(f"Cannot convert {type(obj)} to scalar")


def iter_ranges(
    sel: Union[Scalar, slice, Range, List[Scalar]],
    sparse: bool,
    nonempty_domain: Optional[Range] = None,
) -> Iterator[Range]:
    if isinstance(sel, slice):
        if sel.step is not None:
            raise ValueError("Stepped slice ranges are not supported")

        rstart = sel.start
        if rstart is None and nonempty_domain:
            rstart = nonempty_domain[0]

        rend = sel.stop
        if rend is None and nonempty_domain:
            rend = nonempty_domain[1]

        if sparse and sel.start is None and sel.stop is None:
            # don't set nonempty_domain for full-domain slices w/ sparse
            # because TileDB query is faster without the constraint
            pass
        elif rstart is None or rend is None:
            pass
        else:
            yield to_scalar(rstart), to_scalar(rend)

    elif isinstance(sel, tuple):
        assert len(sel) == 2
        yield to_scalar(sel[0]), to_scalar(sel[1])

    elif isinstance(sel, list):
        for scalar in map(to_scalar, sel):
            yield scalar, scalar

    else:
        scalar = to_scalar(sel)
        yield scalar, scalar


def getitem_ranges(array: Array, idx: Any) -> Sequence[Sequence[Range]]:
    ranges: List[Sequence[Range]] = [()] * array.schema.domain.ndim
    ned = array.nonempty_domain() if array.mode() == "r" else None
    is_sparse = array.schema.sparse
    for i, dim_sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        # don't try to index nonempty_domain if None
        nonempty_domain = ned[i] if ned else None
        if isinstance(dim_sel, np.ndarray):
            ranges[i] = dim_sel
            continue
        elif not isinstance(dim_sel, list):
            dim_sel = [dim_sel]
        ranges[i] = tuple(
            rng
            for sel in dim_sel
            for rng in iter_ranges(sel, is_sparse, nonempty_domain)
        )
    return tuple(ranges)


class Subarray(CtxMixin, lt.Subarray):
    """
    TODO: Documentation for subarray
    """

    def __init__(
        self,
        array: Array,
        ctx: "Ctx" = None,
    ):
        """TODO: Documentation"""
        # Initialize pybind Subarray
        self._array = array
        self._cpp_array = ArrayImpl(array)
        super().__init__(ctx, self._cpp_array)

    def nrange(self, key) -> np.uint64:
        """TODO: Documentation"""
        return self._range_num(key)

    def add_multi_index_ranges(self, indices):
        """TODO: Documentation"""
        if indices is None:
            return

        ranges = getitem_ranges(self._array, indices)
        if hasattr(self, "_add_ranges_bulk") and any(
            isinstance(r, np.ndarray) for r in ranges
        ):
            self._add_ranges_bulk(self.ctx, ranges)
        else:
            self._add_ranges(ranges)

    def add_dim_range(self, dim_idx: int, range: Range) -> None:
        """TODO: Documentation"""
        self._add_dim_range(dim_idx, range)

    def add_label_range(self, label: str, range: Range):
        """TODO: Documentation"""
        self._add_label_range(self, self._ctx, label, range)
