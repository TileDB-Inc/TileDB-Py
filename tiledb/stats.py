from json import dumps as json_dumps
from json import loads as json_loads

from tiledb import TileDBError


def stats_enable():
    """Enable TileDB internal statistics."""
    from .main import init_stats

    init_stats()


def stats_disable():
    """Disable TileDB internal statistics."""
    from .main import disable_stats

    disable_stats()


def stats_reset():
    """Reset all TileDB internal statistics to 0."""
    from .main import reset_stats

    reset_stats()


def stats_dump(
    version=True, print_out=True, include_python=True, json=False, verbose=True
):
    """Return TileDB internal statistics as a string.

    :param include_python: Include TileDB-Py statistics
    :param print_out: Print string to console (default True), or return as string
    :param version: Include TileDB Embedded and TileDB-Py versions (default: True)
    :param json: Return stats JSON object (default: False)
    :param verbose: Print extended internal statistics (default: True)
    """
    from .main import stats_dump_str, stats_enabled, stats_raw_dump_str

    if not stats_enabled():
        raise TileDBError(
            "Statistics are not enabled. Call tiledb.stats_enable() first."
        )

    if json or not verbose:
        stats_str = stats_raw_dump_str()
    else:
        stats_str = stats_dump_str()

    stats_str_core = stats_str.strip()

    if json or not verbose:
        stats_json_core = json_loads(stats_str_core)[0]
        if include_python:
            from .main import python_internal_stats

            stats_json_core["python"] = python_internal_stats(True)
        if json:
            return json_dumps(stats_json_core)

    stats_str = ""

    if version:
        import tiledb

        stats_str += f"TileDB Embedded Version: {tiledb.libtiledb.version()}\n"
        stats_str += f"TileDB-Py Version: {tiledb.version.version}\n"

    if not verbose:
        stats_str += "\n==== READ ====\n\n"

        import tiledb

        if tiledb.libtiledb.version() < (2, 3):
            stats_str += "- Number of read queries: {}\n".format(
                stats_json_core["READ_NUM"]
            )
            stats_str += "- Number of attributes read: {}\n".format(
                stats_json_core["READ_ATTR_FIXED_NUM"]
                + stats_json_core["READ_ATTR_VAR_NUM"]
            )
            stats_str += "- Time to compute estimated result size: {}\n".format(
                stats_json_core["READ_COMPUTE_EST_RESULT_SIZE"]
            )
            stats_str += "- Read time: {}\n".format(stats_json_core["READ"])
            stats_str += (
                "- Total read query time (array open + init state + read): {}\n".format(
                    stats_json_core["READ"] + stats_json_core["READ_INIT_STATE"]
                )
            )
        elif tiledb.libtiledb.version() < (2, 15):
            loop_num = stats_json_core["counters"][
                "Context.StorageManager.Query.Reader.loop_num"
            ]
            stats_str += f"- Number of read queries: {loop_num}\n"

            attr_num = (
                stats_json_core["counters"][
                    "Context.StorageManager.Query.Reader.attr_num"
                ]
                + stats_json_core["counters"][
                    "Context.StorageManager.Query.Reader.attr_fixed_num"
                ]
            )
            stats_str += f"- Number of attributes read: {attr_num}\n"

            read_compute_est_result_size = stats_json_core["timers"].get(
                "Context.StorageManager.Query.Subarray.read_compute_est_result_size.sum"
            )
            if read_compute_est_result_size is not None:
                stats_str += f"- Time to compute estimated result size: {read_compute_est_result_size}\n"

            read_tiles = stats_json_core["timers"][
                "Context.StorageManager.Query.Reader.read_tiles.sum"
            ]
            stats_str += f"- Read time: {read_tiles}\n"

            reads_key = (
                "Context.StorageManager.array_open_READ.sum"
                if tiledb.libtiledb.version() > (2, 15)
                else "Context.StorageManager.array_open_for_reads.sum"
            )

            total_read = (
                stats_json_core["timers"][reads_key]
                + stats_json_core["timers"][
                    "Context.StorageManager.Query.Reader.init_state.sum"
                ]
                + stats_json_core["timers"][
                    "Context.StorageManager.Query.Reader.read_tiles.sum"
                ]
            )
            stats_str += f"- Total read query time (array open + init state + read): {total_read}\n"
    else:
        stats_str += "\n"
        stats_str += stats_str_core
        stats_str += "\n"

    if include_python:
        from .main import python_internal_stats

        stats_str += python_internal_stats()

    if print_out:
        print(stats_str)
    else:
        return stats_str
