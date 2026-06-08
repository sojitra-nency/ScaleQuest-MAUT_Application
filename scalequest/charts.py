"""Altair charts for ScaleQuest.

Replaces the previous figure-based plotting module: no global figure state, no
memory leak, no explicit cleanup needed, plus readable labels and interactive
tooltips.
"""

import altair as alt
from sklearn.preprocessing import MinMaxScaler


def utility_curve_chart(output, title="Utility Curve"):
    """Line+point utility curve: normalized score per option, sorted by score.

    Mutates ``output`` to add a ``normalized_score`` column (Min-Max of
    ``overall_score``), preserving the previous output contract.
    """
    output["normalized_score"] = MinMaxScaler().fit_transform(output[["overall_score"]])

    tooltip = [c for c in ["Vendor", "Abbreviated Vendor", "Company", "overall_score",
                           "normalized_score"] if c in output.columns]
    return (
        alt.Chart(output)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Abbreviated Vendor:N",
                sort=alt.SortField(field="normalized_score", order="descending"),
                axis=alt.Axis(labelAngle=-45, labelOverlap=True, labelLimit=120),
                title="Vendor",
            ),
            y=alt.Y("normalized_score:Q", title="Utility Score"),
            tooltip=tooltip,
        )
        .properties(title=title, height=400)
        .interactive()
    )


def comparison_chart(long_df, top_n=15):
    """Grouped bar of normalized score per method for the top-N options.

    ``long_df`` has columns ``["Abbreviated Vendor", "method", "normalized_score"]``
    and is expected to already be limited to the top-N options by consensus.
    """
    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("normalized_score:Q", title="Normalized score"),
            color=alt.Color("method:N", title="Method"),
            column=alt.Column(
                "Abbreviated Vendor:N",
                title=f"Top {top_n} options by consensus",
                sort=None,
            ),
            tooltip=["Abbreviated Vendor", "method", "normalized_score"],
        )
        .properties(height=300)
    )
