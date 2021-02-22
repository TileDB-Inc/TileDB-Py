# This is a helper function to run tests on an external
# directory, for example the contents of the Pandas
# CSV tests:
#   https://github.com/pandas-dev/pandas/tree/master/pandas/tests/io/data/csv
# It takes one argument, the test directory, and checks that all
# .csv files contained within are correctly round-tripped via
# `tiledb.from_csv` and `tiledb.open_dataframe`

import tiledb
import os
import sys
import tempfile
import pandas as pd
import pandas._testing as tm
from glob import glob


def check_csv_roundtrip(input_csv):
    basename = os.path.basename(input_csv)
    tmp = tempfile.mktemp(prefix="csvtest-" + basename)
    os.mkdir(tmp)

    array_uri = os.path.join(tmp, "tiledb_from_csv")
    tiledb.from_csv(array_uri, input_csv)

    df_csv = pd.read_csv(input_csv)
    df_back = tiledb.open_dataframe(array_uri)

    tm.assert_frame_equal(df_csv, df_back)
    return True


def check_csv_dir(path):
    files = glob(os.path.join(path, "*.csv"))
    res = [check_csv_roundtrip(f) for f in files]

    assert len(res) == len(files), "Failed to check all files!"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("expected one argument: path to CSV directory")

    check_csv_dir(sys.argv[1])
