from os import PathLike
from typing import TYPE_CHECKING
import tiledb

if TYPE_CHECKING:
    import pandas as pd


def from_parquet(uri, parquet_uri) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_parquet(parquet_uri)

    tiledb.from_pandas(uri, df)
