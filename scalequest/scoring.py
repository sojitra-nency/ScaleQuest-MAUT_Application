"""Scoring engines for each MCDA method, plus shared ranking/validity helpers.

This module is intentionally PURE: it imports only numpy/pandas, never
``streamlit`` or ``scalequest.config`` (config imports a couple of helpers from
here at import time, so the dependency must stay one-directional).
"""

import numpy as np

#: Saaty Random Consistency Index, indexed by matrix order n (n=0,1,2 -> 0).
#: Used to normalize the AHP consistency index into a consistency ratio.
DEFAULT_SAATY_RI = [0.0, 0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]


# --- Criteria direction --------------------------------------------------------
def apply_directions(data, direction_map):
    """Invert COST criteria on the already-MinMax-normalized frame, in place.

    For every attribute whose direction is ``"cost"`` we replace ``x`` with
    ``1 - x`` so that, downstream, *higher is always better* for every method.
    Benefit attributes are untouched. Call exactly once, right after
    :func:`scalequest.preprocessing.load_and_preprocess`.
    """
    for attribute, direction in direction_map.items():
        if direction == "cost" and attribute in data.columns:
            data[attribute] = 1.0 - data[attribute]
    return data


# --- Weighting / AHP -----------------------------------------------------------
def ahp_weights(matrix, attributes):
    """Derive AHP priority weights from a pairwise-comparison matrix.

    Uses the geometric-mean (approximate eigenvector) method and normalizes the
    result to sum to 1. Returns ``{attribute: weight}``.
    """
    weights = np.power(np.prod(matrix, axis=1), 1 / matrix.shape[0])
    weights = weights / np.sum(weights)
    return dict(zip(attributes, weights))


def ahp_consistency(matrix, ri_table=DEFAULT_SAATY_RI):
    """Return ``{lambda_max, CI, CR}`` for an AHP pairwise-comparison matrix.

    ``lambda_max = mean((A·w) / w)`` using the geometric-mean weights, then
    ``CI = (lambda_max - n)/(n - 1)`` and ``CR = CI / RI[n]``. A CR <= 0.10 is
    conventionally considered acceptably consistent.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    weights = np.power(np.prod(matrix, axis=1), 1 / n)
    weights = weights / np.sum(weights)

    weighted_sum_vector = matrix @ weights
    lambda_max = float(np.mean(weighted_sum_vector / weights))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = ri_table[n] if n < len(ri_table) else ri_table[-1]
    cr = ci / ri if ri > 0 else 0.0
    return {"lambda_max": lambda_max, "CI": ci, "CR": cr}


# --- Scoring engines (each sets ``overall_score``; higher = better) ------------
def weighted_sum(data, weights):
    """Additive utility (WSM / MANUAL / AHP): each attribute scaled by its weight."""
    for attribute, weight in weights.items():
        data[attribute + "_weighted"] = data[attribute] * weight
    data["overall_score"] = data.filter(like="_weighted").sum(axis=1)
    return data


def loss_score(data, weights, rate=1, transform="exponential"):
    """LOSS utility, weighted and summed.

    ``transform="exponential"`` (default): ``exp(rate * x) * w`` (original behavior).
    ``transform="quadratic"``: ``(x ** 2) * w`` (notebook variant; penalizes
    mediocre scores more sharply).
    """
    for attribute, weight in weights.items():
        if transform == "quadratic":
            data[attribute + "_utility"] = (data[attribute] ** 2) * weight
        else:
            data[attribute + "_utility"] = np.exp(rate * data[attribute]) * weight
    data["overall_score"] = data.filter(like="_utility").sum(axis=1)
    return data


def topsis_score(data, attributes, weights, post_transform="none"):
    """TOPSIS: relative closeness to the ideal-best/ideal-worst solutions.

    ``post_transform`` reshapes the closeness into the utility score:
    ``"none"`` (default) keeps raw closeness; ``"exponential"`` applies
    ``exp(closeness)``; ``"logarithmic"`` applies ``log1p(closeness)``. All three
    are monotonic in closeness, so they change the *curve shape* but never the
    *ranking*.
    """
    weighted = data[attributes] * weights
    ideal_positive = weighted.max()
    ideal_negative = weighted.min()

    positive_distance = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    negative_distance = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))
    closeness = negative_distance / (positive_distance + negative_distance)

    if post_transform == "exponential":
        data["overall_score"] = np.exp(closeness)
    elif post_transform == "logarithmic":
        data["overall_score"] = np.log1p(closeness)
    else:
        data["overall_score"] = closeness
    return data


def weighted_product(data, attributes, weights):
    """WPM (Weighted Product Model): ``score = prod_j (x_j ** w_j)``.

    MinMax-normalized data contains structural zeros (every column's min row is
    0, and cost inversion adds more). A single zero factor collapses the whole
    product to 0, so we first floor the values into ``[0.01, 1]`` via
    ``0.01 + 0.99 * x``. This removes the zeros while preserving the order and
    keeping the best-on-every-criterion option at 1.0. (Plain max-normalization
    is a no-op here because MinMax already makes each column max == 1.)
    """
    sub = data[attributes].to_numpy(dtype=float)
    sub = 0.01 + 0.99 * sub
    w = np.array([weights[a] for a in attributes], dtype=float)
    data["overall_score"] = np.prod(np.power(sub, w), axis=1)
    return data


def vikor_score(data, attributes, weights, v=0.5):
    """VIKOR compromise ranking on benefit-aligned, normalized data.

    Computes S (weighted utility / group regret), R (individual regret), and the
    compromise index Q. Lower Q is better, so we store ``overall_score = -Q`` to
    keep the shared descending sort in :func:`rank_and_select` correct. Also
    exposes raw ``S``, ``R``, ``Q`` columns for inspection.
    """
    X = data[attributes].to_numpy(dtype=float)
    f_star = X.max(axis=0)   # best per criterion (data already benefit-aligned)
    f_minus = X.min(axis=0)  # worst per criterion
    rng = f_star - f_minus
    rng[rng == 0] = 1.0      # constant criterion contributes 0 regret

    w = np.array([weights[a] for a in attributes], dtype=float)
    normalized_regret = w * (f_star - X) / rng
    S = normalized_regret.sum(axis=1)
    R = normalized_regret.max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    dS = (S_minus - S_star) or 1.0
    dR = (R_minus - R_star) or 1.0
    Q = v * (S - S_star) / dS + (1 - v) * (R - R_star) / dR

    data["S"], data["R"], data["Q"] = S, R, Q
    data["overall_score"] = -Q
    return data


def borda_scores(rankings_by_method, n_options):
    """Borda rank aggregation (consensus across methods).

    ``rankings_by_method`` maps method name -> ``{option_id: rank}`` where rank is
    1-based (1 = best). Returns ``{option_id: total_points}`` with
    ``points = sum_over_methods (n_options - rank)``. Higher = stronger consensus.
    """
    totals = {}
    for ranks in rankings_by_method.values():
        for option_id, rank in ranks.items():
            totals[option_id] = totals.get(option_id, 0) + (n_options - rank)
    return totals


# --- Ranking helper ------------------------------------------------------------
def rank_and_select(data, columns):
    """Sort by ``overall_score`` desc, add abbreviated vendor labels, project columns."""
    ranked = data.sort_values("overall_score", ascending=False)
    ranked["Abbreviated Vendor"] = [
        "Vendor {}".format(i + 1) for i in range(len(ranked))
    ]
    return ranked[columns].reset_index(drop=True)
