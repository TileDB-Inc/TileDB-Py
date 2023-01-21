import tiledb


def from_parquet(uri, parquet_uri):
    import pandas as pd

    df = pd.read_parquet(parquet_uri)

    tiledb.from_pandas(uri, df)
