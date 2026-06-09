"""Central configuration: paths, diagrams, and all decision weights.

Every path and every weight/criteria number lives here so the math stays
auditable in one place and no path is re-hardcoded across the app.
"""

from pathlib import Path

import numpy as np

# --- Paths (relative to the project root, robust to the current working dir) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DIAGRAMS_DIR = PROJECT_ROOT / "assets" / "diagrams"

#: Sample dataset shipped with the repo (not auto-loaded; users upload their own).
SAMPLE_DATA_FILE = DATA_DIR / "Consolidated data.xlsx"

#: method -> (concept diagram, "working" diagram)
DIAGRAM_PAIRS = {
    "MANUAL": (
        DIAGRAMS_DIR / "maut_1-MAUT_1.drawio.png",
        DIAGRAMS_DIR / "maut_1-MAUT_1-WORKING.drawio.png",
    ),
    "AHP": (
        DIAGRAMS_DIR / "maut_1-AHP.drawio.png",
        DIAGRAMS_DIR / "maut_1-AHP-WORKING.drawio.png",
    ),
    "LOSS": (
        DIAGRAMS_DIR / "maut_1-LOSS.drawio.png",
        DIAGRAMS_DIR / "maut_1-LOSS-WORKING.drawio.png",
    ),
    "TOPSIS": (
        DIAGRAMS_DIR / "maut_1-TOPSIS.drawio.png",
        DIAGRAMS_DIR / "maut_1-TOPSIS-WORKING.drawio.png",
    ),
}

# --- Decision criteria ---------------------------------------------------------
#: The 9 scored attributes, in the column order used everywhere downstream.
ATTRIBUTES = [
    "IRR",
    "Strategic fit",
    "Technical Feasibility",
    "Uniqueness of R&D",
    "Reputational risk",
    "Market and Business risk",
    "Scalability",
    "Regulatory risk",
    "Market factors",
]

#: Direction of each criterion: benefit (maximize) or cost (minimize). The three
#: risk attributes are costs; everything else is a benefit. Cost criteria are
#: inverted after normalization so that higher is always better downstream.
CRITERIA_DIRECTION = {
    "IRR": "benefit",
    "Strategic fit": "benefit",
    "Technical Feasibility": "benefit",
    "Uniqueness of R&D": "benefit",
    "Reputational risk": "cost",
    "Market and Business risk": "cost",
    "Scalability": "benefit",
    "Regulatory risk": "cost",
    "Market factors": "benefit",
}

#: Fixed weights for the MANUAL weighted-sum (also the base for the sensitivity sweep).
MANUAL_WEIGHTS = {
    "IRR": 0.2,
    "Strategic fit": 0.1,
    "Technical Feasibility": 0.15,
    "Uniqueness of R&D": 0.1,
    "Reputational risk": 0.1,
    "Market and Business risk": 0.1,
    "Scalability": 0.1,
    "Regulatory risk": 0.1,
    "Market factors": 0.05,
}

#: AHP / LOSS pairwise-comparison matrix (rows/cols aligned with ATTRIBUTES).
#: Weights are derived from this via the geometric-mean method.
AHP_MATRIX = np.array([
    [1, 3, 5, 3, 7, 9, 5, 7, 3],          # IRR
    [1/3, 1, 3, 3, 7, 7, 5, 7, 3],        # Strategic fit
    [1/5, 1/3, 1, 1/3, 3, 3, 3, 5, 1],    # Technical Feasibility
    [1/3, 1/3, 3, 1, 5, 7, 5, 7, 3],      # Uniqueness of R&D
    [1/7, 1/7, 1/3, 1/5, 1, 3, 3, 5, 1],  # Reputational risk
    [1/9, 1/7, 1/3, 1/7, 1/3, 1, 1/3, 1, 1/3],  # Market and Business risk
    [1/5, 1/5, 1/3, 1/5, 1/3, 3, 1, 3, 1/3],    # Scalability
    [1/7, 1/7, 1/5, 1/7, 1/5, 1, 1/3, 1, 1/5],  # Regulatory risk
    [1/3, 1/3, 1, 1/3, 1, 3, 3, 5, 1],    # Market factors
])

#: Rate of increase in the LOSS exponential utility function.
LOSS_RATE = 1

#: IRR weights tested by the sensitivity analysis — 8 points from near-zero to 0.50.
SENSITIVITY_IRR_RANGE = np.linspace(0.05, 0.50, 8).round(3).tolist()

# --- Method tuning -------------------------------------------------------------
#: VIKOR strategy weight (0.5 = balance group utility and individual regret).
VIKOR_V = 0.5

#: AHP consistency ratio threshold; above this the pairwise matrix is "inconsistent".
CR_THRESHOLD = 0.10

# --- Output column selections --------------------------------------------------
RANK_COLUMNS = ["S.no", "Company", "Vendor", "Abbreviated Vendor", "overall_score"]
#: TOPSIS_RANK_COLUMNS deliberately omits the full "Vendor" name to reduce table width
#: and avoid clutter in the TOPSIS ranked output (Company is sufficient for identification).
TOPSIS_RANK_COLUMNS = ["S.no", "Company", "Abbreviated Vendor", "overall_score"]

# --- Precomputed AHP values (matrix is static, so compute once at import) -------
# Imported here from the pure scoring module; scoring must NOT import config back.
from scalequest.scoring import (  # noqa: E402  (deliberate bottom import)
    DEFAULT_SAATY_RI,
    ahp_consistency,
    ahp_weights,
)

#: Saaty Random Index table, surfaced from scoring for any UI that wants it.
SAATY_RI = DEFAULT_SAATY_RI

#: AHP priority weights {attribute: weight} derived from AHP_MATRIX.
AHP_WEIGHTS = ahp_weights(AHP_MATRIX, ATTRIBUTES)

#: AHP consistency {lambda_max, CI, CR} for AHP_MATRIX.
AHP_CONSISTENCY = ahp_consistency(AHP_MATRIX)
