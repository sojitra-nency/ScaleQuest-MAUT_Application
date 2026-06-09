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

    if data.empty:
        return None  # signal to the caller

    data.columns = (
        ["S.no", "Company", "Vendor"]
        + config.ATTRIBUTES
        + data.columns[12:].tolist()
    )

    data[config.ATTRIBUTES] = data[config.ATTRIBUTES].apply(
        pd.to_numeric, errors="coerce"
    )

    # Check for NaN after coercion and warn the user
    nan_mask = data[config.ATTRIBUTES].isna()
    if nan_mask.any().any():
        bad_cols = nan_mask.any(axis=0)
        bad_col_names = [c for c, has_nan in zip(config.ATTRIBUTES, bad_cols) if has_nan]
        st.warning(f"Non-numeric values found in: {', '.join(bad_col_names)} — affected cells scored as 0.")

    data[config.ATTRIBUTES] = MinMaxScaler().fit_transform(data[config.ATTRIBUTES])
    return data


def load_and_preprocess(file):
    """Read an uploaded ``.xlsx`` (or path) and return a normalized DataFrame.

    Returns a fresh ``.copy()`` because callers mutate the frame in place (adding
    weighted/utility columns); without the copy a cache hit would hand back a
    frame already polluted by a previous method run.
    """
    if hasattr(file, "getvalue"):
        data_bytes = file.getvalue()
    else:
        with open(file, "rb") as f:
            data_bytes = f.read()

    data = _preprocess_bytes(data_bytes)

    if data is None or data.empty:
        st.error("The uploaded file produced an empty dataset after processing. Check the file format.")
        st.stop()

    return data.copy()
