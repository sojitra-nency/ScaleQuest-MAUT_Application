"""The MCDA methods: each renders its diagrams (if any), ranked table, and chart.

Every method shares the signature ``(file, weights, use_costs)`` so ``app.py`` can
dispatch uniformly. Weight-based methods (MANUAL, TOPSIS, VIKOR, WPM, SENSITIVITY)
consume ``weights``; AHP and LOSS use the pairwise-matrix weights and ignore it.
"""

import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from scalequest import config
from scalequest.charts import comparison_chart, utility_curve_chart
from scalequest.preprocessing import load_and_preprocess
from scalequest.scoring import (
    apply_directions,
    borda_scores,
    loss_score,
    rank_and_select,
    topsis_score,
    vikor_score,
    weighted_product,
    weighted_sum,
)


def _show_diagrams(method):
    """Render the concept + 'working' diagrams, or nothing if the method has none."""
    pair = config.DIAGRAM_PAIRS.get(method)
    if pair is None:
        return
    concept, working = pair
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(concept))
    with col2:
        st.image(str(working))


def _prepare(file, use_costs):
    """Load (cached) + optionally invert cost criteria. One copy per call."""
    data = load_and_preprocess(file)
    if use_costs:
        data = apply_directions(data, config.CRITERIA_DIRECTION)
    return data


def _render(output, title="Utility Curve"):
    st.write(output)
    st.altair_chart(utility_curve_chart(output, title), use_container_width=True)


def _ahp_consistency_note():
    cr = config.AHP_CONSISTENCY["CR"]
    if cr > config.CR_THRESHOLD:
        st.warning(
            f"AHP pairwise-matrix consistency ratio CR = {cr:.3f} exceeds "
            f"{config.CR_THRESHOLD:.2f} — the derived weights may be unreliable."
        )
    else:
        st.caption(f"AHP consistency ratio CR = {cr:.3f} (≤ {config.CR_THRESHOLD:.2f}, acceptable).")
    st.caption(
        "AHP and LOSS use weights derived from the pairwise matrix; "
        "the sidebar weight sliders do not apply to these two methods."
    )


def manual(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("MAUT = Multi Attribute Utility Theory")
    _show_diagrams("MANUAL")

    data = weighted_sum(_prepare(file, use_costs), weights)
    _render(rank_and_select(data, config.RANK_COLUMNS), "Utility Curve")


def ahp(file, weights=None, use_costs=True):
    st.title("AHP = Analytical Hierarchy Process")
    _show_diagrams("AHP")
    _ahp_consistency_note()

    data = weighted_sum(_prepare(file, use_costs), config.AHP_WEIGHTS)
    _render(rank_and_select(data, config.RANK_COLUMNS), "Utility Curve")


def loss(file, weights=None, use_costs=True):
    st.title("LOSS function = Level of Service Satisfaction")
    _show_diagrams("LOSS")
    _ahp_consistency_note()

    transform = st.radio(
        "Utility transform", ["exponential", "quadratic"], horizontal=True,
        help="exponential: exp(x)·w (default). quadratic: x²·w (penalizes mediocre scores).",
    )
    data = loss_score(
        _prepare(file, use_costs), config.AHP_WEIGHTS, rate=config.LOSS_RATE, transform=transform
    )
    _render(rank_and_select(data, config.RANK_COLUMNS), "Utility Curve")


def topsis(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("TOPSIS = Technique for Order Preference by Similarity to Ideal Solution.")
    _show_diagrams("TOPSIS")

    post_transform = st.radio(
        "Post-transform on closeness", ["none", "exponential", "logarithmic"], horizontal=True,
        help="All three are monotonic in closeness, so they reshape the curve but do NOT change the ranking.",
    )
    weight_list = [weights[a] for a in config.ATTRIBUTES]
    data = topsis_score(_prepare(file, use_costs), config.ATTRIBUTES, weight_list, post_transform=post_transform)
    _render(rank_and_select(data, config.TOPSIS_RANK_COLUMNS), "Utility Curve")


def vikor(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("VIKOR = Compromise ranking (VlseKriterijumska Optimizacija)")
    st.caption("Ranks by the compromise index Q (lower Q = better; shown here as a utility curve).")

    data = vikor_score(_prepare(file, use_costs), config.ATTRIBUTES, weights, v=config.VIKOR_V)
    _render(rank_and_select(data, config.RANK_COLUMNS), "Utility Curve")


def wpm(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("WPM = Weighted Product Model")
    st.caption("Score = ∏ xⱼ^wⱼ (criteria max-normalized to avoid structural zeros).")

    data = weighted_product(_prepare(file, use_costs), config.ATTRIBUTES, weights)
    _render(rank_and_select(data, config.RANK_COLUMNS), "Utility Curve")


def sensitivity(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("Sensitivity Test — IRR weight sweep")

    data = _prepare(file, use_costs)
    for weight in config.SENSITIVITY_IRR_RANGE:
        swept = dict(weights)
        swept["IRR"] = weight
        scored = weighted_sum(data.copy(), swept)
        output = rank_and_select(scored, config.RANK_COLUMNS)
        st.write(output)
        st.altair_chart(
            utility_curve_chart(output, title=f"Utility Curve (IRR weight = {weight})"),
            use_container_width=True,
        )


def compare(file, weights=None, use_costs=True):
    weights = weights or config.MANUAL_WEIGHTS
    st.title("COMPARE — all methods side by side")
    st.caption("Each method's score is Min-Max normalized to [0, 1]; consensus is a Borda count over ranks.")

    base = _prepare(file, use_costs)
    topsis_weights = [weights[a] for a in config.ATTRIBUTES]
    runners = {
        "MANUAL": lambda d: weighted_sum(d, weights),
        "AHP": lambda d: weighted_sum(d, config.AHP_WEIGHTS),
        "LOSS": lambda d: loss_score(d, config.AHP_WEIGHTS, rate=config.LOSS_RATE),
        "TOPSIS": lambda d: topsis_score(d, config.ATTRIBUTES, topsis_weights),
        "VIKOR": lambda d: vikor_score(d, config.ATTRIBUTES, weights, v=config.VIKOR_V),
        "WPM": lambda d: weighted_product(d, config.ATTRIBUTES, weights),
    }

    result = base[["S.no", "Company", "Vendor"]].copy()
    rankings_by_method = {}
    for name, fn in runners.items():
        scored = fn(base.copy())[["S.no", "overall_score"]].copy()
        scored["score"] = MinMaxScaler().fit_transform(scored[["overall_score"]])
        scored["rank"] = scored["overall_score"].rank(ascending=False, method="min").astype(int)
        rankings_by_method[name] = dict(zip(scored["S.no"], scored["rank"]))
        result = result.merge(
            scored[["S.no", "score", "rank"]].rename(
                columns={"score": f"{name} score", "rank": f"{name} rank"}
            ),
            on="S.no",
        )

    borda = borda_scores(rankings_by_method, len(result))
    result["Consensus (Borda)"] = result["S.no"].map(borda)
    result = result.sort_values("Consensus (Borda)", ascending=False).reset_index(drop=True)
    result.insert(0, "Abbreviated Vendor",
        [f"#{i + 1} {v[:14]}" for i, v in enumerate(result["Vendor"])])

    st.dataframe(result, use_container_width=True)

    top_n = min(15, len(result))
    top = result.head(top_n)
    score_cols = [f"{m} score" for m in runners]
    long_df = top.melt(
        id_vars=["Abbreviated Vendor"], value_vars=score_cols,
        var_name="method", value_name="normalized_score",
    )
    long_df["method"] = long_df["method"].str.replace(" score", "", regex=False)
    st.altair_chart(comparison_chart(long_df, top_n=top_n), use_container_width=True)
