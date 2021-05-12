from contextlib import contextmanager
from contextvars import ContextVar

import tiledb

_ctx_var = ContextVar("ctx")


@contextmanager
def scope_ctx(ctx_or_config=None):
    """
    Context manager for setting the default `tiledb.Ctx` context variable when entering
    a block of code and restoring it to its previous value when exiting the block.

    :param ctx_or_config: :py:class:`tiledb.Ctx` or :py:class:`tiledb.Config` object
        or dictionary with config parameters.
    :return: Ctx
    """
    if not isinstance(ctx_or_config, tiledb.Ctx):
        ctx = tiledb.Ctx(ctx_or_config)
    else:
        ctx = ctx_or_config
    token = _ctx_var.set(ctx)
    try:
        yield _ctx_var.get()
    finally:
        _ctx_var.reset(token)


def default_ctx(config=None):
    """
    Returns, and optionally initializes, the default `tiledb.Ctx` context variable.

    This Ctx object is used by Python API functions when no `ctx` keyword argument
    is provided. Most API functions accept an optional `ctx` kwarg, but that is typically
    only necessary in advanced usage with multiple contexts per program.

    For initialization, this function must be called before any other tiledb functions.
    The initialization call accepts a  :py:class:`tiledb.Config` object to override the
    defaults for process-global parameters.

    :param config: :py:class:`tiledb.Config` object or dictionary with config parameters.
    :return: Ctx
    """
    try:
        ctx = _ctx_var.get()
        if config is not None:
            raise tiledb.TileDBError("Global context already initialized!")
    except LookupError:
        ctx = tiledb.Ctx(config)
        _ctx_var.set(ctx)
    return ctx
