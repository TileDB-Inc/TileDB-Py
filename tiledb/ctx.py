import io
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Union

import tiledb
import tiledb.cc as lt

_ctx_var = ContextVar("ctx")

already_warned = False
_needs_fork_wrapper = sys.platform != "win32" and sys.version_info < (3, 12)


class Config(lt.Config):
    """TileDB Config class

    The Config object stores configuration parameters for both TileDB Embedded
    and TileDB-Py.

    For TileDB Embedded parameters, see:

        https://docs.tiledb.com/main/how-to/configuration#configuration-parameters

    The following configuration options are supported by TileDB-Py:

        - `py.init_buffer_bytes`:

           Initial allocation size in bytes for attribute and dimensions buffers.
           If result size exceed the pre-allocated buffer(s), then the query will return
           incomplete and TileDB-Py will allocate larger buffers and resubmit.
           Specifying a sufficiently large buffer size will often improve performance.
           Default 10 MB (1024**2 * 10).

        - `py.use_arrow`:

           Use `pyarrow` from the Apache Arrow project to convert
           query results into Pandas dataframe format when requested.
           Default `True`.

        - `py.deduplicate`:

           Attempt to deduplicate Python objects during buffer
           conversion to Python. Deduplication may reduce memory usage for datasets
           with many identical strings, at the cost of some performance reduction
           due to hash calculation/lookup for each object.

    Unknown parameters will be ignored!

    :param dict params: Set parameter values from dict like object
    :param str path: Set parameter values from persisted Config parameter file
    """

    def __init__(self, params: dict = None, path: str = None):
        super().__init__()
        if path is not None:
            self.load(path)
        if params is not None:
            self.update(params)

    @staticmethod
    def load(uri: str):
        """Constructs a Config class instance from config parameters loaded from a local Config file

        :parameter str uri: a local URI config file path
        :rtype: tiledb.Config
        :return: A TileDB Config instance with persisted parameter values
        :raises TypeError: `uri` cannot be converted to a unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return lt.Config(uri)

    def __setitem__(self, key: str, value: str):
        """Sets a config parameter value.

        :param str key: Name of parameter to set
        :param str value: Value of parameter to set
        :raises TypeError: `key` or `value` cannot be encoded into a UTF-8 string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        self.set(str(key), str(value))

    def get(self, key: str, raise_keyerror: bool = True):
        try:
            return super().get(key)
        except Exception:
            if raise_keyerror:
                raise KeyError(key)
            else:
                return None

    def __getitem__(self, key: str):
        """Gets a config parameter value.

        :param str key: Name of parameter to get
        :return: Config parameter value string
        :rtype str:
        :raises TypeError: `key` cannot be encoded into a UTF-8 string
        :raises KeyError: Config parameter not found
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self.get(key, True)

    def __delitem__(self, key: str):
        """
        Removes a configured parameter (resetting it to its default).

        :param str key: Name of parameter to reset.
        :raises TypeError: `key` cannot be encoded into a UTF-8 string

        """
        self.unset(key)

    def __iter__(self):
        """Returns an iterator over the Config parameters (keys)"""
        return ConfigKeys(self)

    def __len__(self):
        """Returns the number of parameters (keys) held by the Config object"""
        return sum(1 for _ in self)

    def __eq__(self, config):
        if not isinstance(config, Config):
            return False
        keys = set(self.keys())
        okeys = set(config.keys())
        if keys != okeys:
            return False
        for k in keys:
            val, oval = self[k], config[k]
            if val != oval:
                return False
        return True

    def __repr__(self):
        colnames = ["Parameter", "Value"]
        params = list(self.keys())
        values = list(map(repr, self.values()))
        colsizes = [
            max(len(colnames[0]), *map(len, (p for p in params))),
            max(len(colnames[1]), *map(len, (v for v in values))),
        ]
        format_str = " | ".join("{{:<{}}}".format(i) for i in colsizes)
        output = []
        output.append(format_str.format(colnames[0], colnames[1]))
        output.append(format_str.format("-" * colsizes[0], "-" * colsizes[1]))
        output.extend(format_str.format(p, v) for p, v in zip(params, values))
        return "\n".join(output)

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")

        output.write("<tr>")
        output.write("<th>Parameter</th>")
        output.write("<th>Value</th>")
        output.write("</tr>")

        params = list(self.keys())
        values = list(map(repr, self.values()))

        for p, v in zip(params, values):
            output.write("<tr>")
            output.write(f"<td>{p}</td>")
            output.write(f"<td>{v}</td>")
            output.write("</tr>")

        output.write("</table>")

        return output.getvalue()

    def items(self, prefix: str = ""):
        """Returns an iterator object over Config parameters, values

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigItems
        :returns: iterator over Config parameter, value tuples

        """
        return ConfigItems(self, prefix=prefix)

    def keys(self, prefix: str = ""):
        """Returns an iterator object over Config parameters (keys)

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigKeys
        :returns: iterator over Config parameter string keys

        """
        return ConfigKeys(self, prefix=prefix)

    def values(self, prefix: str = ""):
        """Returns an iterator object over Config values

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigValues
        :returns: iterator over Config string values

        """
        return ConfigValues(self, prefix=prefix)

    def dict(self, prefix: str = ""):
        """Returns a dict representation of a Config object

        :param str prefix: return only parameters with a given prefix
        :rtype: dict
        :return: Config parameter / values as a Python dict

        """
        return dict(ConfigItems(self, prefix=prefix))

    def clear(self):
        """Unsets all Config parameters (returns them to their default values)"""
        for k in self.keys():
            del self[k]

    # def get(self, key, *args: Optional[str]):
    #     """Gets the value of a config parameter, or a default value.

    #     :param str key: Config parameter
    #     :param args: return `arg` if Config does not contain parameter `key`
    #     :return: Parameter value, `arg` or None.

    #     """
    #     nargs = len(args)
    #     if nargs > 1:
    #         raise TypeError("get expected at most 2 arguments, got {}".format(nargs))
    #     try:
    #         return self[key]
    #     except KeyError:
    #         return args[0] if nargs == 1 else None

    def update(self, odict: dict):
        """Update a config object with parameter, values from a dict like object

        :param odict: dict-like object containing parameter, values to update Config.

        """
        super().update(dict(odict))

    def from_file(self, path: str):
        """Update a Config object with from a persisted config file

        :param path: A local Config file path

        """
        config = Config.load(path)
        self.update(config)

    def save(self, uri: str):
        """Persist Config parameter values to a config file

        :parameter str uri: a local URI config file path
        :raises TypeError: `uri` cannot be converted to a unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        self.save_to_file(uri)


class ConfigKeys:
    """
    An iterator object over Config parameter strings (keys)
    """

    def __init__(self, config: Config, prefix: str = ""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (k, _) = self.config_items.__next__()
        return k


class ConfigValues:
    """
    An iterator object over Config parameter value strings
    """

    def __init__(self, config: Config, prefix: str = ""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (_, v) = self.config_items.__next__()
        return v


class ConfigItems:
    """
    An iterator object over Config parameter, values

    :param config: TileDB Config object
    :type config: tiledb.Config
    :param prefix: (default "") Filter paramter names with given prefix
    :type prefix: str

    """

    def __init__(self, config: Config, prefix: str = ""):
        self.config = config
        self.iter = config._iter(prefix)

    def __iter__(self):
        return self.iter

    def __next__(self):
        return self.iter.__next__()


class Ctx(lt.Context):
    """Class representing a TileDB context.

    A TileDB context wraps a TileDB storage manager.

    :param config: Initialize Ctx with given config parameters
    :type config: tiledb.Config or dict

    """

    def __init__(self, config: Config = None):
        _config = lt.Config()

        if config is not None:
            if isinstance(config, lt.Config):
                _config = config
            elif isinstance(config, Config):
                _config.update(config.dict())
            elif isinstance(config, dict):
                _config.update(config)
            else:
                raise TypeError(
                    "Ctx's config argument expects type `tiledb.Config` or `dict`"
                )

        super().__init__(_config)

        self._set_default_tags()

        # The core tiledb library uses threads and it's easy
        # to experience deadlocks when forking a process that is using
        # tiledb.  The project doesn't have a solution for this at the
        # moment other than to avoid using fork(), which is the same
        # recommendation that Python makes. Python 3.12 warns if you
        # fork() when multiple threads are detected and Python 3.14 will
        # make it so you never accidentally fork(): multiprocessing will
        # default to "spawn" on Linux.
        _ensure_os_fork_wrap()

    def __repr__(self):
        return "tiledb.Ctx() [see Ctx.config() for configuration]"

    def config(self):
        """Returns the Config instance associated with the Ctx."""
        new = Config.__new__(Config)
        # bypass calling Config.__init__, call lt.Config.__init__ instead
        lt.Config.__init__(new, super().config())
        return new

    def set_tag(self, key: str, value: str):
        """Sets a (string, string) "tag" on the Ctx (internal)."""
        super().set_tag(key, value)

    def _set_default_tags(self):
        """Sets all default tags on the Ctx"""
        self.set_tag("x-tiledb-api-language", "python")
        self.set_tag(
            "x-tiledb-api-language-version",
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}",
        )
        self.set_tag("x-tiledb-api-sys-platform", sys.platform)

    def get_stats(self, print_out: bool = True, json: bool = False):
        """Retrieves the stats from a TileDB context.

        :param print_out: Print string to console (default True), or return as string
        :param json: Return stats JSON object (default: False)
        """
        stats = super().get_stats()

        if json:
            import json

            output = json.loads(stats)
        else:
            output = stats

        if print_out:
            print(output)
        else:
            return output


class CtxMixin:
    """
    Base mixin class for pure Python classes that extend PyBind11 TileDB classes.

    To use this class, a subclass must:
    - Inherit from it first (i.e. `class Foo(CtxMixin, Bar)`, not `class Foo(Bar, CtxMixin)`
    - Call super().__init__ by passing `ctx` (tiledb.Ctx or None) as first parameter and
      zero or more pure Python positional parameters
    """

    def __init__(self, ctx, *args, _pass_ctx_to_super=True):
        if not ctx:
            ctx = default_ctx()

        if _pass_ctx_to_super:
            super().__init__(ctx, *args)
        else:
            super().__init__(*args)

        # we set this here because if the super().__init__() constructor above fails,
        # we don't want to set self._ctx
        self._ctx = ctx

    @classmethod
    def from_capsule(cls, ctx, capsule):
        """Create an instance of this class from a PyCapsule instance"""
        # bypass calling self.__init__, call CtxMixin.__init__ instead
        self = cls.__new__(cls)
        CtxMixin.__init__(self, ctx, capsule)
        return self

    @classmethod
    def from_pybind11(cls, ctx, lt_obj):
        """Create an instance of this class from a PyBind11 instance"""
        # bypass calling self.__init__, call CtxMixin.__init__ instead
        self = cls.__new__(cls)
        CtxMixin.__init__(self, ctx, lt_obj, _pass_ctx_to_super=False)
        return self


def check_ipykernel_warn_once():
    """
    This function checks if we have imported ipykernel version < 6 in the
    current process, and provides a warning that default_ctx/scope_ctx will
    not work correctly due to a bug in IPython contextvar support."""
    global already_warned
    if not already_warned:
        try:
            import warnings

            if "ipykernel" in sys.modules and tuple(
                map(int, sys.modules["ipykernel"].__version__.split("."))
            ) < (6, 0):
                warnings.warn(
                    "tiledb.default_ctx and scope_ctx will not function correctly "
                    "due to bug in IPython contextvar support.  You must supply a "
                    "Ctx object to each function for custom configuration options. "
                    "Please consider upgrading to ipykernel >= 6!"
                    "Please see https://github.com/TileDB-Inc/TileDB-Py/issues/667 "
                    "for more information."
                )
        except:
            pass
        finally:
            already_warned = True


@contextmanager
def scope_ctx(ctx_or_config: Union["Ctx", "Config", dict] = None) -> "Ctx":
    """
    Context manager for setting the default `tiledb.Ctx` context variable when entering
    a block of code and restoring it to its previous value when exiting the block.

    :param ctx_or_config: :py:class:`tiledb.Ctx` or :py:class:`tiledb.Config` object
        or dictionary with config parameters.
    :return: Ctx
    """
    check_ipykernel_warn_once()

    if ctx_or_config is not None and not (
        isinstance(ctx_or_config, tiledb.Ctx)
        or isinstance(ctx_or_config, tiledb.Config)
        or isinstance(ctx_or_config, dict)
    ):
        raise ValueError(
            "scope_ctx takes in `tiledb.Ctx` object, `tiledb.Config` object, or "
            "dictionary with config parameters."
        )

    if not isinstance(ctx_or_config, tiledb.Ctx):
        ctx = tiledb.Ctx(ctx_or_config)
    else:
        ctx = ctx_or_config
    token = _ctx_var.set(ctx)
    try:
        yield _ctx_var.get()
    finally:
        _ctx_var.reset(token)


def default_ctx(config: Union["Config", dict] = None) -> "Ctx":
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
    check_ipykernel_warn_once()

    if config is not None and not (
        isinstance(config, tiledb.Config) or isinstance(config, dict)
    ):
        raise ValueError(
            "default_ctx takes in `tiledb.Config` object or "
            "dictionary with config parameters."
        )

    try:
        ctx = _ctx_var.get()
        if config is not None:
            raise tiledb.TileDBError("Global context already initialized!")

        # The core tiledb library uses threads and it's easy
        # to experience deadlocks when forking a process that is using
        # tiledb.  The project doesn't have a solution for this at the
        # moment other than to avoid using fork(), which is the same
        # recommendation that Python makes. Python 3.12 warns if you
        # fork() when multiple threads are detected and Python 3.14 will
        # make it so you never accidentally fork(): multiprocessing will
        # default to "spawn" on Linux.
        _ensure_os_fork_wrap()
    except LookupError:
        ctx = tiledb.Ctx(config)
        _ctx_var.set(ctx)
    return ctx


def _ensure_os_fork_wrap():
    global _needs_fork_wrapper
    if _needs_fork_wrapper:
        import os
        import warnings
        from functools import wraps

        def warning_wrapper(func):
            @wraps(func)
            def wrapper():
                warnings.warn(
                    "TileDB is a multithreading library and deadlocks "
                    "are likely if fork() is called after a TileDB "
                    "context has been created (such as for array "
                    "access). To safely use TileDB with "
                    "multiprocessing or concurrent.futures, choose "
                    "'spawn' as the start method for child processes. "
                    "For example: "
                    "multiprocessing.set_start_method('spawn').",
                    UserWarning,
                )
                return func()

            return wrapper

        os.fork = warning_wrapper(os.fork)
        _needs_fork_wrapper = False
