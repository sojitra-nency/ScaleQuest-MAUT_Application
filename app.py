"""ScaleQuest — Streamlit entry point.

Pick a MAUT method in the sidebar, upload an Excel dataset, and the selected
method ranks the options and plots a utility curve. All scoring lives in the
``scalequest`` package; this file only wires the UI to it.
"""

import streamlit as st

from scalequest.methods import ahp, loss, manual, sensitivity, topsis

# Sidebar label -> method function. A dict makes routing impossible to get wrong.
DISPATCH = {
    "MANUAL": manual,
    "AHP": ahp,
    "LOSS": loss,
    "TOPSIS": topsis,
    "SENSITIVITY_TEST": sensitivity,
}

option = st.sidebar.selectbox("Select an approach", list(DISPATCH.keys()))
upload_file = st.file_uploader("Upload an Excel (.xlsx) file", type=["xlsx"])

if upload_file is not None:
    DISPATCH[option](upload_file)
else:
    st.info("Upload an .xlsx dataset to run the selected analysis.")
