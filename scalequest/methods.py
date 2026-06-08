"""The five MAUT methods, each rendering its diagrams, ranked table and curve."""

import streamlit as st

from scalequest import config
from scalequest.plotting import plot_utility_curve
from scalequest.preprocessing import load_and_preprocess
from scalequest.scoring import (
    ahp_weights,
    loss_score,
    rank_and_select,
    topsis_score,
    weighted_sum,
)


def _show_diagrams(method):
    """Render the concept and 'working' diagrams for a method side by side."""
    concept, working = config.DIAGRAM_PAIRS[method]
    col1, col2 = st.columns([2, 2])
    with col1:
        st.image(str(concept))
    with col2:
        st.image(str(working))


def manual(file):
    st.title("MAUT = Multi Attribute Utility Theory")
    _show_diagrams("MANUAL")

    data = load_and_preprocess(file)
    data = weighted_sum(data, config.MANUAL_WEIGHTS)
    output = rank_and_select(data, config.RANK_COLUMNS)

    st.write(output)
    st.pyplot(plot_utility_curve(output))


def ahp(file):
    st.title("AHP = Analytical Hierarchy Process")
    _show_diagrams("AHP")

    data = load_and_preprocess(file)
    weights = ahp_weights(config.AHP_MATRIX, config.ATTRIBUTES)
    data = weighted_sum(data, weights)
    output = rank_and_select(data, config.RANK_COLUMNS)

    st.write(output)
    st.pyplot(plot_utility_curve(output))


def loss(file):
    st.title("LOSS function = Level of Service Satisfaction")
    _show_diagrams("LOSS")

    data = load_and_preprocess(file)
    weights = ahp_weights(config.AHP_MATRIX, config.ATTRIBUTES)
    data = loss_score(data, weights, rate=config.LOSS_RATE)
    output = rank_and_select(data, config.RANK_COLUMNS)

    st.write(output)
    st.pyplot(plot_utility_curve(output))


def topsis(file):
    st.title("TOPSIS = Technique for Order Preference by Similarity to Ideal Solution.")
    _show_diagrams("TOPSIS")

    data = load_and_preprocess(file)
    data = topsis_score(data, config.ATTRIBUTES, config.TOPSIS_WEIGHTS)
    output = rank_and_select(data, config.TOPSIS_RANK_COLUMNS)

    st.write(output)
    st.pyplot(plot_utility_curve(output))


def sensitivity(file):
    st.title("Sensitivity Test — IRR weight sweep")

    data = load_and_preprocess(file)
    for weight in config.SENSITIVITY_IRR_RANGE:
        weights = dict(config.MANUAL_WEIGHTS)
        weights["IRR"] = weight

        scored = weighted_sum(data.copy(), weights)
        output = rank_and_select(scored, config.RANK_COLUMNS)

        st.write(output)
        st.pyplot(
            plot_utility_curve(
                output, title="Utility Curve (Weight for IRR: {})".format(weight)
            )
        )
