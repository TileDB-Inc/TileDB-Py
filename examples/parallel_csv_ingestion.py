# parallel_csv_ingestion.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2020 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# DESCRIPTION
#
# This example demonstrates ingestion of CSV files in parallel
# with tiledb.from_csv and Python multiprocessing.
#

import glob
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import tiledb

# helper functions to generate data
from tiledb.tests.common import rand_datetime64_array, rand_utf8

# are we running as a test
in_test = "PYTEST_CURRENT_TEST" in os.environ


def check_dataframe_deps():
    pd_error = """Pandas version >= 1.0 and < 3.0 required for dataframe functionality.
                  Please `pip install pandas>=1.0,<3.0` to proceed."""

    try:
        import pandas as pd
    except ImportError:
        raise Exception(pd_error)

    from packaging.version import Version

    if Version(pd.__version__) < Version("1.0") or Version(pd.__version__) >= Version(
        "3.0.0.dev0"
    ):
        raise Exception(pd_error)


def generate_csvs(csv_folder, count=9, min_length=1, max_length=109):
    def make_dataframe(col_size):
        data = {
            "idx_datetime": rand_datetime64_array(col_size, include_extremes=False),
            "column_int64": np.random.randint(0, 150000, size=col_size, dtype=np.int64),
            "column_uint32": np.random.randint(
                0, 150000, size=col_size, dtype=np.uint32
            ),
            "column_float64": np.random.rand(col_size),
            "column_utf8": np.array(
                [rand_utf8(np.random.randint(1, 100)) for _ in range(col_size)]
            ),
        }
        df = pd.DataFrame.from_dict(data)

        df.set_index("idx_datetime", inplace=True)
        return df

    # create list of CSV row-counts to generate
    # (each file will have nrows from this list)
    csv_lengths = np.random.randint(min_length, max_length, size=count)

    for i, target_length in enumerate(csv_lengths):
        output_path = os.path.join(csv_folder, "gen_csv_{}.csv".format(i))

        df = make_dataframe(target_length)
        df.to_csv(output_path)


def log_process_errors(*args, **kwargs):
    try:
        tiledb.from_csv(*args, **kwargs)
    except Exception as exc:
        # print log to file. randomize just in case
        err_id = np.random.randint(np.iinfo(np.int64).max - 1)
        err_filename = f"ingest-err-PID_{os.getpid()}_{err_id}.log"
        print("err_filename: ", err_filename)
        err = f"""              ------------------------
              Caught exception:
              ------------------------
              {exc}
              ------------------------
              with args:
              ------------------------
              {args}
              ------------------------
              with kwargs:
              ------------------------
              {kwargs}
              ------------------------
              this message saved to file: {err_filename}
              """
        print(err)

        with open(err_filename, "w") as f:
            f.writelines(err)

        raise exc


def from_csv_mp(
    csv_path,
    array_path,
    list_step_size=5,
    chunksize=100,
    max_workers=4,
    initial_file_count=5,
    index_col=None,
    parse_dates=True,
    attr_types=None,
    sparse=True,
    allows_duplicates=True,
    debug=False,
    **kwargs,
):
    """
    Multi-process ingestion wrapper around tiledb.from_csv

    Currently uses ProcessPoolExecutor.
    """

    # Setting start method to 'spawn' is required to
    # avoid problems with process global state when spawning via fork.
    # NOTE: *must be inside __main__* or a function.
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    # Get a list of of CSVs from the target path
    csvs = glob.glob(csv_path + "/*.csv")

    if len(csvs) < 1:
        raise ValueError("Cannot ingest empty CSV list!")

    # first step: create the array. we read the first N csvs to create schema
    #             and as check for inconsistency before starting the full run.
    tiledb.from_csv(
        array_path,
        csvs[:initial_file_count],
        chunksize=chunksize,  # must set chunksize here even though schema_only
        index_col=index_col,
        parse_dates=parse_dates,
        dtype=attr_types,
        column_types=attr_types,
        engine="c",
        debug=debug,
        allows_duplicates=True,
        sparse=sparse,
        mode="schema_only",
        **kwargs,
    )

    print("Finished array schema creation")

    # controls number of CSV files passed to each worker process:
    # depending on the makeup of the files, we may want to read a number of
    # files consecutively (up to chunksize) in order to write more optimal
    # fragments.
    if list_step_size > len(csvs):
        raise ValueError(
            "Please choose a step size smaller than the number of CSV files"
        )

    tasks = []
    # high level ingestion timing
    start = time.time()

    # ingest the data in parallel

    # note: use ThreadPoolExecutor for debugging
    #       use ProcessPoolExecutor in general
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for first in range(0, len(csvs), list_step_size):
            last = min(len(csvs), first + list_step_size)
            print("  Submitting task for CSV list range: ", (first, last))
            task = executor.submit(
                log_process_errors,
                *(array_path, csvs[first:last]),
                **dict(
                    chunksize=chunksize,
                    index_col=index_col,
                    parse_dates=parse_dates,
                    dtype=attr_types,
                    column_types=attr_types,
                    engine="c",
                    debug=debug,
                    allows_duplicates=allows_duplicates,
                ),
                **kwargs,
                mode="append",
            )
            tasks.append(task)

    print("Task results: ", [t.result() for t in tasks])

    print("Ingestion complete. Duration: ", time.time() - start)


##############################################################################
# Usage example
##############################################################################
def example():
    # set up test paths and data
    csv_path = tempfile.mkdtemp()
    generate_csvs(csv_path, count=11)
    print("Finished generating CSVs in path: ", csv_path)

    array_path = tempfile.mkdtemp()
    print("Writing output array to: ", array_path)

    # Create Schema
    attr_types = {
        "column_int64": np.int64,
        "column_uint32": np.uint32,
        "column_float64": np.float64,
        "column_utf8": str,
    }

    from_csv_mp(
        csv_path,
        array_path,
        chunksize=27,
        list_step_size=5,
        max_workers=4,
        index_col=["idx_datetime"],
        attr_types=attr_types,
    )

    print("Ingestion complete.")
    print("  Note: temp paths have undefined lifetime after exit.")

    # apparently no good way to check for "is interactive" in python
    if not in_test:
        input("  Press any key to continue: ")

    return csv_path, array_path


if __name__ == "__main__" and not in_test:
    example()


##############################################################################
# TEST SECTION
# uses this example as a test of various input combinations
##############################################################################
def df_from_csvs(path, **kwargs):
    idx_column = kwargs.pop("tiledb_idx_column", None)

    csv_paths = glob.glob(path + "/*.csv")
    csv_df_list = [pd.read_csv(p, **kwargs) for p in csv_paths]

    df = pd.concat(csv_df_list)

    if idx_column is not None:
        df.sort_values(idx_column, inplace=True)
        df.set_index(idx_column, inplace=True)
        df.index = df.index.astype("datetime64[ns]")

    return df


def test_parallel_csv_ingestion():
    csv_path, array_path = example()
    import pandas._testing as tm

    attr_types = {
        "column_int64": np.int64,
        "column_uint32": np.uint32,
        "column_float64": np.float64,
        # Avoid this runtime warning: "DeprecationWarning: `np.str` is a deprecated alias for the builtin `str`."
        "column_utf8": str,
    }

    # read dataframe from CSV list, set index, and sort
    df_direct = df_from_csvs(
        csv_path, dtype=attr_types, tiledb_idx_column="idx_datetime"
    )

    # validate the array generated in example()
    df_tiledb = tiledb.open_dataframe(array_path)
    tm.assert_frame_equal(df_direct, df_tiledb.sort_values("idx_datetime"))

    # ingest over several parameters
    for nproc in [1, 5]:  # note: already did 4 above
        for csv_list_step in [5, 11]:
            for chunksize in [10, 100]:
                array_tmp = tempfile.mkdtemp()

                print(
                    "Running ingestion with nproc: '{}', step: '{}', chunksize: '{}'".format(
                        nproc, csv_list_step, chunksize
                    )
                )
                print("Writing output array to: ", array_tmp)

                from_csv_mp(
                    csv_path,
                    array_tmp,
                    chunksize=chunksize,
                    list_step_size=csv_list_step,
                    max_workers=nproc,
                    index_col=["idx_datetime"],
                    attr_types=attr_types,
                )

                df_tiledb = tiledb.open_dataframe(array_tmp)
                tm.assert_frame_equal(df_direct, df_tiledb.sort_values("idx_datetime"))

    print("Writing output array to: ", array_path)


if __name__ == "__main__":
    check_dataframe_deps()
    import pandas as pd

    test_parallel_csv_ingestion()
