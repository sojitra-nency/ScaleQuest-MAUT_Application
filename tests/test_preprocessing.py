"""Tests for the preprocessing layer against the shipped sample dataset.

``load_and_preprocess`` accepts a path, so these run headless — the bad-data
branches (which call ``st.warning``/``st.error``) are not triggered by the
well-formed sample file.
"""

import pytest

from scalequest import config
from scalequest.preprocessing import load_and_preprocess


@pytest.fixture(scope="module")
def sample():
    return load_and_preprocess(config.SAMPLE_DATA_FILE)


def test_sample_loads_expected_shape(sample):
    assert len(sample) == 140
    for col in ("S.no", "Company", "Vendor"):
        assert col in sample.columns
    for attr in config.ATTRIBUTES:
        assert attr in sample.columns


def test_attributes_are_normalized_floats(sample):
    attrs = sample[config.ATTRIBUTES]
    assert (attrs.dtypes == "float64").all()
    assert attrs.min().min() >= 0.0
    assert attrs.max().max() <= 1.0
    assert not attrs.isna().any().any()  # NaN must not survive preprocessing


def test_cache_copy_isolation():
    # Mutating one returned frame must not poison the next load (cache returns .copy()).
    first = load_and_preprocess(config.SAMPLE_DATA_FILE)
    first["IRR"] = 999.0
    second = load_and_preprocess(config.SAMPLE_DATA_FILE)
    assert second["IRR"].max() <= 1.0
