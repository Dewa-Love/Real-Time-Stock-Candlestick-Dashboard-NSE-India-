import pandas as pd
import numpy as np

def fix_ohlc(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            series = df[col]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            series = series.apply(
                lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
            )

            df[col] = pd.to_numeric(series, errors="coerce")

    df = df.dropna()
    return df
