import numpy as np
from typing import Any, Tuple, TYPE_CHECKING

import tiledb.cc as lt
from .ctx import default_ctx
from .np2buf import dtype_to_tiledb
from .util import dtype_range, tiledb_cast_tile_extent, tiledb_type_is_datetime

if TYPE_CHECKING:
    from .libtiledb import Ctx


class DimensionLabel(lt.DimensionLabel):
    def __init__(
        self,
        domain: Tuple[Any, Any] = None,
        tile: Any = None,
        order: str = "increasing",
        original_dtype: np.dtype = np.uint64,
        new_dtype: np.dtype = np.uint64,
        ctx: "Ctx" = None,
    ):
        ctx = ctx or default_ctx()

        if domain is not None and len(domain) != 2:
            raise ValueError("invalid domain extent, must be a pair")

        if (isinstance(new_dtype, str) and new_dtype == "ascii") or np.dtype(
            new_dtype
        ).kind == "S":
            # Handle var-len dom type (currently only TILEDB_STRING_ASCII)
            # The dims's dom is implicitly formed as coordinates are written.
            new_tdb_dtype = lt.DataType.STRING_ASCII
        else:
            if new_dtype is not None:
                new_dtype = np.dtype(new_dtype)
                new_dtype_min, new_dtype_max = dtype_range(new_dtype)

                if domain == (None, None):
                    # this means to use the full extent of the type
                    domain = (new_dtype_min, new_dtype_max)
                elif (
                    domain[0] < new_dtype_min
                    or domain[0] > new_dtype_max
                    or domain[1] < new_dtype_min
                    or domain[1] > new_dtype_max
                ):
                    raise TypeError(
                        "invalid domain extent, domain cannot be safely"
                        f" cast to dtype {new_dtype!r}"
                    )

            domain_buffer = np.asarray(domain, dtype=new_dtype)
            domain_dtype = domain_buffer.dtype
            new_tdb_dtype = dtype_to_tiledb(domain_dtype)

            # check that the domain type is a valid dtype (integer / floating)
            if (
                not np.issubdtype(domain_dtype, np.integer)
                and not np.issubdtype(domain_dtype, np.floating)
                and not domain_dtype.kind == "M"
            ):
                raise TypeError(f"invalid Dim dtype {domain_dtype!r}")

            if tiledb_type_is_datetime(new_tdb_dtype):
                domain_buffer = domain_buffer.astype(dtype=np.int64)

            # if the tile extent is specified, cast
            if tile is not None:
                tiledb_buffer = tiledb_cast_tile_extent(tile, domain_dtype)
                if tiledb_buffer.size != 1:
                    raise ValueError("tile extent must be a scalar")

        original_tdb_dtype = dtype_to_tiledb(original_dtype)
        str_to_order = {
            "increasing": lt.LabelOrder.INCREASING_LABELS,
            "decreasing": lt.LabelOrder.DECREASING_LABELS,
            "unordered": lt.LabelOrder.UNORDERED_LABELS,
        }
        label_order = str_to_order[order]

        super().__init__(
            ctx,
            label_order,
            original_tdb_dtype,
            new_tdb_dtype,
            domain_buffer,
            tiledb_buffer,
        )
