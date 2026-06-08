"""Shared data loading and normalization for every MAUT method."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from scalequest import config


def load_and_preprocess(file):
    """Read an uploaded ``.xlsx`` and return a normalized DataFrame.

    Steps (identical for every method): skip the title row, drop the leading
    index column and the two trailing summary rows, name the first 12 columns,
    coerce the 9 attribute columns to numeric, and Min-Max scale them to [0, 1].

    ``file`` may be a path or a Streamlit ``UploadedFile`` (any file-like
    object accepted by :func:`pandas.read_excel`).
    """
    data = pd.read_excel(file, skiprows=1)
    data = data.iloc[:, 1:]      # drop the leading index column
    data = data.iloc[:-2]        # drop the two trailing summary rows

    data.columns = (
        ["S.no", "Company", "Vendor"]
        + config.ATTRIBUTES
        + data.columns[12:].tolist()
    )

    data[config.ATTRIBUTES] = data[config.ATTRIBUTES].apply(
        pd.to_numeric, errors="coerce"
    )
    data[config.ATTRIBUTES] = MinMaxScaler().fit_transform(data[config.ATTRIBUTES])
    return data
