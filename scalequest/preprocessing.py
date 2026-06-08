"""Shared data loading and normalization for every MCDA method.

The heavy step (Excel parse + Min-Max fit) is cached keyed on the file's raw
bytes, so switching methods, dragging weight sliders, or toggling options reuse
the cached frame instead of re-parsing. Uploading a different file changes the
bytes and busts the cache.
"""

import io

import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from scalequest import config


@st.cache_data(show_spinner=False)
def _preprocess_bytes(data_bytes):
    """Cached core. Keyed on raw bytes (hashable, content-addressed)."""
    data = pd.read_excel(io.BytesIO(data_bytes), skiprows=1)
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


def load_and_preprocess(file):
    """Read an uploaded ``.xlsx`` (or path) and return a normalized DataFrame.

    Returns a fresh ``.copy()`` because callers mutate the frame in place (adding
    ``*_weighted``/``*_utility`` columns); without the copy a cache hit would hand
    back a frame already polluted by a previous method run.
    """
    data_bytes = file.getvalue() if hasattr(file, "getvalue") else open(file, "rb").read()
    return _preprocess_bytes(data_bytes).copy()
