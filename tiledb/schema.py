from .ctx import default_ctx
from .libtiledb import ArraySchema, Attr, Dim, Domain

import numpy as np


def regularize_tiling(tile, ndim):
    if not tile:
        return None
    elif np.isscalar(tile):
        tiling = tuple(int(tile) for _ in range(ndim))
    elif (tile is str) or (len(tile) != ndim):
        raise ValueError(
            "'tile' argument must be iterable " "and match array dimensionality"
        )
    else:
        tiling = tuple(tile)
    return tiling


def schema_like_numpy(array, ctx=None, **kw):
    """create array schema from Numpy array-like object
    internal function. tiledb.schema_like is exported and recommended
    """
    if not ctx:
        ctx = default_ctx()
    # create an ArraySchema from the numpy array object
    tiling = regularize_tiling(kw.pop("tile", None), array.ndim)

    attr_name = kw.pop("attr_name", "")
    dim_dtype = kw.pop("dim_dtype", np.uint64)
    dims = []
    for (dim_num, d) in enumerate(range(array.ndim)):
        # support smaller tile extents by kw
        # domain is based on full shape
        tile_extent = tiling[d] if tiling else array.shape[d]
        domain = (0, array.shape[d] - 1)
        dims.append(Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx))

    var = False
    if array.dtype == object:
        # for object arrays, we use the dtype of the first element
        # consistency check should be done later, if needed
        el0 = array.flat[0]
        if type(el0) is bytes:
            el_dtype = np.dtype("S")
            var = True
        elif type(el0) is str:
            el_dtype = np.dtype("U")
            var = True
        elif type(el0) == np.ndarray:
            if len(el0.shape) != 1:
                raise TypeError(
                    "Unsupported sub-array type for Attribute: {} "
                    "(only string arrays and 1D homogeneous NumPy arrays are supported)".format(
                        type(el0)
                    )
                )
            el_dtype = el0.dtype
        else:
            raise TypeError(
                "Unsupported sub-array type for Attribute: {} "
                "(only strings and homogeneous-typed NumPy arrays are supported)".format(
                    type(el0)
                )
            )
    else:
        el_dtype = array.dtype

    att = Attr(dtype=el_dtype, name=attr_name, var=var, ctx=ctx)
    dom = Domain(*dims, ctx=ctx)
    return ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)


def schema_like(*args, shape=None, dtype=None, ctx=None, **kw):
    """
    Return an ArraySchema corresponding to a NumPy-like object or
    `shape` and `dtype` kwargs. Users are encouraged to pass 'tile'
    and 'capacity' keyword arguments as appropriate for a given
    application.

    :param A: NumPy array-like object, or TileDB reference URI, optional
    :param tuple shape: array shape, optional
    :param dtype: array dtype, optional
    :param Ctx ctx: TileDB Ctx
    :param kwargs: additional keyword arguments to pass through, optional
    :return: tiledb.ArraySchema
    """
    if not ctx:
        ctx = default_ctx()

    def is_ndarray_like(obj):
        return hasattr(arr, "shape") and hasattr(arr, "dtype") and hasattr(arr, "ndim")

    # support override of default dimension dtype
    dim_dtype = kw.pop("dim_dtype", np.uint64)
    if len(args) == 1:
        arr = args[0]
        if is_ndarray_like(arr):
            tiling = regularize_tiling(kw.pop("tile", None), arr.ndim)
            schema = schema_like_numpy(arr, tile=tiling, dim_dtype=dim_dtype, ctx=ctx)
        else:
            raise ValueError("expected ndarray-like object")
    elif shape and dtype:
        if np.issubdtype(np.bytes_, dtype):
            dtype = np.dtype("S")
        elif np.issubdtype(dtype, np.unicode_):
            dtype = np.dtype("U")

        ndim = len(shape)
        tiling = regularize_tiling(kw.pop("tile", None), ndim)

        dims = []
        for d in range(ndim):
            # support smaller tile extents by kw
            # domain is based on full shape
            tile_extent = tiling[d] if tiling else shape[d]
            domain = (0, shape[d] - 1)
            dims.append(Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx))

        att = Attr(dtype=dtype, ctx=ctx)
        dom = Domain(*dims, ctx=ctx)
        schema = ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)
    elif kw is not None:
        raise ValueError
    else:
        raise ValueError(
            "Must provide either ndarray-like object or 'shape' "
            "and 'dtype' keyword arguments"
        )

    return schema
