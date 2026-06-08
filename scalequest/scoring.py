"""Scoring engines for each MAUT method, plus a shared ranking helper."""

import numpy as np


def weighted_sum(data, weights):
    """Additive utility: each attribute scaled by its weight, then summed.

    Used by MANUAL (fixed weights) and AHP (geometric-mean weights).
    """
    for attribute, weight in weights.items():
        data[attribute + "_weighted"] = data[attribute] * weight
    data["overall_score"] = data.filter(like="_weighted").sum(axis=1)
    return data


def ahp_weights(matrix, attributes):
    """Derive AHP priority weights from a pairwise-comparison matrix.

    Uses the geometric-mean (approximate eigenvector) method and normalizes the
    result to sum to 1. Returns ``{attribute: weight}``.
    """
    weights = np.power(np.prod(matrix, axis=1), 1 / matrix.shape[0])
    weights = weights / np.sum(weights)
    return dict(zip(attributes, weights))


def loss_score(data, weights, rate=1):
    """LOSS: exponential utility ``exp(rate * x)`` scaled by each weight."""
    for attribute, weight in weights.items():
        data[attribute + "_utility"] = np.exp(rate * data[attribute]) * weight
    data["overall_score"] = data.filter(like="_utility").sum(axis=1)
    return data


def topsis_score(data, attributes, weights):
    """TOPSIS: relative closeness to the ideal-best/ideal-worst solutions."""
    weighted = data[attributes] * weights
    ideal_positive = weighted.max()
    ideal_negative = weighted.min()

    positive_distance = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    negative_distance = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))

    data["overall_score"] = negative_distance / (positive_distance + negative_distance)
    return data


def rank_and_select(data, columns):
    """Sort by ``overall_score`` desc, add abbreviated vendor labels, project columns.

    Replaces the old write-CSV-then-read-it-back round trip with an in-memory
    selection. ``columns`` is per-method so each table keeps its own shape.
    """
    ranked = data.sort_values("overall_score", ascending=False)
    ranked["Abbreviated Vendor"] = [
        "Vendor {}".format(i + 1) for i in range(len(ranked))
    ]
    return ranked[columns].reset_index(drop=True)
