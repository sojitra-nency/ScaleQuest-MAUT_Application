"""ScaleQuest — Streamlit entry point.

Pick an MCDA method in the sidebar, optionally edit the criterion weights and the
cost/benefit toggle, upload an Excel dataset, and the selected method ranks the
options and plots a utility curve. All scoring lives in the ``scalequest`` package.
"""

import streamlit as st

from scalequest import config
from scalequest.methods import (
    ahp,
    compare,
    loss,
    manual,
    sensitivity,
    topsis,
    vikor,
    wpm,
)

# Sidebar label -> method function. A dict makes routing impossible to get wrong.
DISPATCH = {
    "MANUAL": manual,
    "AHP": ahp,
    "LOSS": loss,
    "TOPSIS": topsis,
    "VIKOR": vikor,
    "WPM": wpm,
    "SENSITIVITY_TEST": sensitivity,
    "COMPARE (all methods)": compare,
}


def _weight_panel():
    """Sidebar sliders for the 9 criterion weights, normalized to sum 1."""
    st.sidebar.subheader("Criterion weights")
    st.sidebar.caption("Used by MANUAL, TOPSIS, VIKOR, WPM, and the sensitivity base.")
    raw = {
        attr: st.sidebar.slider(attr, 0.0, 1.0, float(config.MANUAL_WEIGHTS[attr]), 0.01)
        for attr in config.ATTRIBUTES
    }
    total = sum(raw.values())
    if total == 0:
        st.sidebar.warning("All weights are 0 — falling back to equal weights.")
        n = len(config.ATTRIBUTES)
        return {attr: 1.0 / n for attr in config.ATTRIBUTES}
    normalized = {attr: value / total for attr, value in raw.items()}
    if abs(total - 1.0) > 0.005:
        st.sidebar.caption(
            f"Slider sum = {total:.2f} — normalized to 100%. Effective: "
            + " · ".join(f"{a[:4]}={v:.0%}" for a, v in normalized.items())
        )
    return normalized


option = st.sidebar.selectbox("Select an approach", list(DISPATCH.keys()))
use_costs = st.sidebar.checkbox("Treat risk attributes as costs (recommended)", value=True)
weights = _weight_panel()

upload_file = st.file_uploader("Upload an Excel (.xlsx) file", type=["xlsx"])

if upload_file is not None:
    DISPATCH[option](upload_file, weights, use_costs)
else:
    st.info("Upload an .xlsx dataset to run the selected analysis.")
