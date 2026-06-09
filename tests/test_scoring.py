"""Unit tests for the pure scoring layer (scalequest.scoring).

These functions take plain DataFrames/arrays and return DataFrames/dicts with no
Streamlit or config dependency, so every case here is hand-checkable.
"""

import numpy as np
import pandas as pd
import pytest

from scalequest import config
from scalequest.scoring import (
    ahp_consistency,
    ahp_weights,
    apply_directions,
    borda_scores,
    loss_score,
    rank_and_select,
    topsis_score,
    vikor_score,
    weighted_product,
    weighted_sum,
)


# --- apply_directions ----------------------------------------------------------
def test_apply_directions_inverts_cost_leaves_benefit():
    data = pd.DataFrame({"cost_attr": [0.0, 0.5, 1.0], "benefit_attr": [0.0, 0.5, 1.0]})
    out = apply_directions(data, {"cost_attr": "cost", "benefit_attr": "benefit"})
    assert out["cost_attr"].tolist() == [1.0, 0.5, 0.0]      # inverted
    assert out["benefit_attr"].tolist() == [0.0, 0.5, 1.0]   # untouched


def test_apply_directions_ignores_missing_columns():
    data = pd.DataFrame({"present": [0.2, 0.8]})
    out = apply_directions(data, {"absent": "cost", "present": "benefit"})
    assert out["present"].tolist() == [0.2, 0.8]


# --- ahp_weights ---------------------------------------------------------------
def test_ahp_weights_sum_to_one():
    weights = ahp_weights(config.AHP_MATRIX, config.ATTRIBUTES)
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


def test_ahp_weights_irr_is_largest():
    # The matrix is constructed so IRR dominates; it should carry the most weight.
    weights = ahp_weights(config.AHP_MATRIX, config.ATTRIBUTES)
    assert weights["IRR"] == max(weights.values())


# --- ahp_consistency -----------------------------------------------------------
def test_ahp_consistency_on_project_matrix():
    result = ahp_consistency(config.AHP_MATRIX)
    assert result["CR"] == pytest.approx(0.064, abs=0.005)
    assert result["lambda_max"] == pytest.approx(9.74, abs=0.05)


def test_ahp_consistency_perfect_2x2_is_zero():
    # A perfectly consistent 2x2 matrix has CR == 0.
    matrix = np.array([[1.0, 2.0], [0.5, 1.0]])
    result = ahp_consistency(matrix)
    assert result["CR"] == pytest.approx(0.0, abs=1e-9)


# --- weighted_sum --------------------------------------------------------------
def test_weighted_sum_matches_dot_product():
    data = pd.DataFrame({"A": [0.5, 0.6], "B": [0.3, 0.4]})
    out = weighted_sum(data.copy(), {"A": 0.6, "B": 0.4})
    assert out["overall_score"].tolist() == pytest.approx([0.5 * 0.6 + 0.3 * 0.4,
                                                           0.6 * 0.6 + 0.4 * 0.4])


def test_weighted_sum_leaves_no_intermediate_columns():
    data = pd.DataFrame({"A": [0.5, 0.6], "B": [0.3, 0.4]})
    out = weighted_sum(data.copy(), {"A": 0.6, "B": 0.4})
    assert not any("_weighted" in c for c in out.columns)
    assert set(out.columns) == {"A", "B", "overall_score"}


# --- loss_score ----------------------------------------------------------------
def test_loss_score_exponential_and_quadratic_differ():
    data = pd.DataFrame({"A": [0.5, 0.9], "B": [0.3, 0.4]})
    weights = {"A": 0.6, "B": 0.4}
    exp_out = loss_score(data.copy(), weights, transform="exponential")["overall_score"]
    quad_out = loss_score(data.copy(), weights, transform="quadratic")["overall_score"]
    assert not np.allclose(exp_out, quad_out)
    assert np.isfinite(exp_out).all() and np.isfinite(quad_out).all()


def test_loss_score_no_intermediate_columns():
    data = pd.DataFrame({"A": [0.5, 0.6], "B": [0.3, 0.4]})
    out = loss_score(data.copy(), {"A": 0.6, "B": 0.4})
    assert not any("_utility" in c for c in out.columns)


# --- topsis_score --------------------------------------------------------------
def test_topsis_identical_rows_give_half():
    # When every row is identical, both distances are 0; the guard returns 0.5.
    data = pd.DataFrame({"X": [0.5, 0.5, 0.5], "Y": [0.5, 0.5, 0.5]})
    out = topsis_score(data.copy(), ["X", "Y"], [0.5, 0.5])
    assert not out["overall_score"].isna().any()
    assert (out["overall_score"] == 0.5).all()


def test_topsis_dominant_row_ranks_first():
    data = pd.DataFrame({"X": [0.9, 0.5, 0.1], "Y": [0.9, 0.5, 0.1]})
    out = topsis_score(data.copy(), ["X", "Y"], [0.5, 0.5])
    assert out["overall_score"].idxmax() == 0  # the dominant row


def test_topsis_post_transforms_preserve_ranking():
    data = pd.DataFrame({"X": [0.9, 0.5, 0.1], "Y": [0.2, 0.8, 0.4]})
    base = topsis_score(data.copy(), ["X", "Y"], [0.5, 0.5], post_transform="none")
    order_none = base["overall_score"].rank().tolist()
    for transform in ("exponential", "logarithmic"):
        out = topsis_score(data.copy(), ["X", "Y"], [0.5, 0.5], post_transform=transform)
        assert out["overall_score"].rank().tolist() == order_none


# --- weighted_product ----------------------------------------------------------
def test_weighted_product_zero_entry_does_not_collapse():
    # A raw 0 entry must NOT force the whole product to 0 (the floor handles it).
    data = pd.DataFrame({"X": [0.0, 0.5, 1.0], "Y": [1.0, 0.5, 1.0]})
    out = weighted_product(data.copy(), ["X", "Y"], {"X": 0.5, "Y": 0.5})
    assert (out["overall_score"] > 0).all()


def test_weighted_product_dominant_option_wins():
    data = pd.DataFrame({"X": [1.0, 0.5, 0.1], "Y": [1.0, 0.5, 0.1]})
    out = weighted_product(data.copy(), ["X", "Y"], {"X": 0.5, "Y": 0.5})
    assert out["overall_score"].idxmax() == 0


# --- vikor_score ---------------------------------------------------------------
def test_vikor_lowest_q_has_highest_score():
    data = pd.DataFrame({"X": [0.9, 0.5, 0.1], "Y": [0.8, 0.5, 0.2]})
    out = vikor_score(data.copy(), ["X", "Y"], {"X": 0.5, "Y": 0.5})
    # overall_score = -Q, so the min-Q row must have the max overall_score.
    assert out["overall_score"].idxmax() == out["Q"].idxmin()


def test_vikor_constant_criterion_no_nan():
    data = pd.DataFrame({"X": [0.5, 0.6, 0.4], "Y": [1.0, 1.0, 1.0]})
    out = vikor_score(data.copy(), ["X", "Y"], {"X": 0.7, "Y": 0.3})
    assert not out["Q"].isna().any()


# --- borda_scores --------------------------------------------------------------
def test_borda_scores_match_hand_calculation():
    # 3 options, 2 methods. points = sum over methods of (N - rank).
    rankings = {
        "M1": {1: 1, 2: 2, 3: 3},   # option1 best
        "M2": {1: 2, 2: 1, 3: 3},   # option2 best
    }
    result = borda_scores(rankings, n_options=3)
    # option1: (3-1)+(3-2)=3 ; option2: (3-2)+(3-1)=3 ; option3: 0+0=0
    assert result == {1: 3, 2: 3, 3: 0}


# --- rank_and_select -----------------------------------------------------------
def test_rank_and_select_sorts_and_labels():
    data = pd.DataFrame({
        "S.no": [1, 2, 3],
        "overall_score": [0.2, 0.9, 0.5],
    })
    out = rank_and_select(data, ["S.no", "Abbreviated Vendor", "overall_score"])
    assert out["overall_score"].tolist() == [0.9, 0.5, 0.2]            # descending
    assert out["Abbreviated Vendor"].tolist() == ["Vendor 1", "Vendor 2", "Vendor 3"]
    assert out["S.no"].tolist() == [2, 3, 1]                           # follows sort
