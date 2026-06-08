"""Utility-curve plotting shared by every method."""

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_utility_curve(output, title="Utility Curve"):
    """Min-Max scale the overall scores and plot the utility curve.

    Returns a Matplotlib ``Figure`` for the caller to hand to ``st.pyplot``.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    output["normalized_score"] = scaler.fit_transform(output[["overall_score"]])

    fig, ax = plt.subplots()
    ax.plot(output["Abbreviated Vendor"], output["normalized_score"])
    ax.set_xlabel("Vendor")
    ax.set_ylabel("Utility Score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=2)
    return fig
