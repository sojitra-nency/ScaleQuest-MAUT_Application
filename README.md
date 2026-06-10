# ScaleQuest — MCDA Ranking Tool for Mining R&D

**ScaleQuest** is a decision-support web application that ranks options across multiple weighted criteria using eight Multi-Criteria Decision Analysis (MCDA) methods. Built with Streamlit, it's designed to evaluate ~140 mining & resource R&D vendors across 9 evaluation criteria, but is flexible for any scoring scenario.

**Live demo:** [https://maut-1234.streamlit.app/](https://sojitra-nency-scalequest-maut-application-app-ybo7ns.streamlit.app/)

---

## What ScaleQuest Does

1. **Upload** an Excel spreadsheet with options (vendors) and their scores on 9 criteria
2. **Select** an analysis method from the sidebar (MANUAL, AHP, LOSS, TOPSIS, VIKOR, WPM, SENSITIVITY, COMPARE)
3. **Configure** weights via sliders and choose whether to treat risk criteria as costs
4. **Rank** all options, view normalized scores, and visualize utility curves

The app normalizes all scores to [0, 1], applies cost inversion to risk criteria, derives or accepts weights, and returns a sorted ranking with interactive Altair charts.

---

## The 9 Evaluation Criteria

| Criterion | Type | Direction | Notes |
|-----------|------|-----------|-------|
| **IRR** | Benefit | Maximize | Internal Rate of Return — financial return potential |
| **Strategic fit** | Benefit | Maximize | Alignment with company strategy and goals |
| **Technical Feasibility** | Benefit | Maximize | Likelihood of technical success |
| **Uniqueness of R&D** | Benefit | Maximize | IP differentiation and competitive advantage |
| **Reputational risk** | Cost | Minimize | Inverted to benefit after normalization |
| **Market & Business risk** | Cost | Minimize | Inverted to benefit after normalization |
| **Scalability** | Benefit | Maximize | Ability to scale post-development |
| **Regulatory risk** | Cost | Minimize | Inverted to benefit after normalization |
| **Market factors** | Benefit | Maximize | Size and growth of target market |

**Cost inversion:** The three risk criteria (Reputational, Market & Business, Regulatory) are minimization objectives. After Min-Max normalization, ScaleQuest inverts them via `1 − x` so that all methods work with *higher-is-better* semantics.

---

## Data Processing Pipeline

1. **Upload** an `.xlsx` file
2. **Parse** with `pd.read_excel(..., skiprows=1)` — assumes a header row to skip
3. **Extract columns**: drop leading index column, drop 2 trailing summary rows
4. **Rename** first 12 columns as `["S.no", "Company", "Vendor"]` + the 9 attributes
5. **Coerce to numeric**: `pd.to_numeric(..., errors="coerce")` — non-numeric cells become NaN and are shown in a warning
6. **Normalize**: Min-Max scale each attribute to [0, 1]
7. **Invert costs**: if the "Invert cost criteria" toggle is on, compute `1 − cost_column` for the 3 risk attributes
8. **Cache**: the normalized frame is cached by file content (bytes), so switching methods or dragging sliders reuses the frame

---

## The 8 MCDA Methods

### 1. MANUAL (Weighted Sum Model)

**What it is:** Standard additive utility function using fixed hand-set weights.

**Formula:**
```
Score(i) = Σ(w_j · x_{i,j})
```
where w_j are weights and x_{i,j} are normalized attribute values for option i.

**How to read:** Higher score = better overall fit. Uses the weight sliders in the sidebar.

**When to use:** Baseline linear model; easy to explain and audit.

---

### 2. AHP (Analytic Hierarchy Process)

**What it is:** Derives weights from a pairwise-comparison matrix, then applies weighted sum.

**Weights derived from:** A 9×9 pairwise-comparison matrix (shown below). Uses the geometric-mean method:
```
w_j = (∏ a_{j,k})^(1/n) / Σ weights
```

**Consistency check:** AHP consistency ratio (CR) = 0.064 for this matrix (acceptable; CR ≤ 0.10 is the threshold).

**How to read:** Same as MANUAL, but weights come from the fixed pairwise matrix, not sliders. Sidebar sliders are ignored.

**When to use:** Systematic weight elicitation when you have pairwise preference data; ignores the weight sliders.

---

### 3. LOSS (Level of Service Satisfaction)

**What it is:** Applies an exponential (or quadratic) utility transformation to AHP weights.

**Formula (exponential, default):**
```
Score(i) = Σ(exp(rate · x_{i,j}) · w_j)
```
where rate = 1 (set in config). Exponential emphasizes diminishing returns.

**Formula (quadratic variant):**
```
Score(i) = Σ(x_{i,j}² · w_j)
```

**Toggle:** Radio button in the app lets you switch between exponential and quadratic.

**How to read:** Higher score = better. Quadratic variant penalizes low scores more sharply (middle-ground options score worse).

**When to use:** Captures non-linear satisfaction; useful when mediocre performance is heavily penalized.

---

### 4. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

**What it is:** Ranks options by their relative distance to the ideal-best and ideal-worst solutions.

**Formula:**
```
Closeness(i) = d⁻(i) / (d⁺(i) + d⁻(i))
```
where d⁻ = distance to worst, d⁺ = distance to best, both weighted and Euclidean.

**Post-transform options:** Radio button allows monotone reshaping of closeness:
- "none" (default): raw closeness
- "exponential": exp(closeness)
- "logarithmic": log(1 + closeness)

All three preserve ranking; they only reshape the curve.

**How to read:** Closeness ∈ [0, 1]; higher = closer to ideal, farther from worst.

**When to use:** When you want to penalize options that are weak on any dimension (distance-based ranking).

---

### 5. VIKOR (VlseKriterijumska Optimizacija — Multicriteria Optimization)

**What it is:** Compromise ranking based on majority-rule and individual-regret indices.

**Formula:**
```
Q(i) = v · (S(i) − S*) / (S⁻ − S*) + (1 − v) · (R(i) − R*) / (R⁻ − R*)
```
where:
- S(i) = group utility (sum of weighted regrets)
- R(i) = maximum individual regret
- v = 0.5 (balance between group and individual)
- S*, S⁻, R*, R⁻ = best/worst S and R values

Lower Q = better compromise.

**Display note:** The Y-axis shows `−Q` (negated), so higher on the chart = lower Q = better.

**How to read:** Ranks by the compromise index; options near 1 are close to the ideal, near 0 are far.

**When to use:** When you want a compromise that avoids extreme trades (majority rule + minority protection).

---

### 6. WPM (Weighted Product Model)

**What it is:** Multiplicative model that avoids the structural-zero problem via a floor.

**Formula:**
```
Score(i) = ∏(x_{i,j}^{w_j})
```
where x_{i,j} ∈ [0.01, 1] (floored from [0, 1] to avoid structural zeros).

**Why the floor:** Min-Max normalization makes each column's minimum 0. A single zero factor would collapse the product to zero for all options. The fix `0.01 + 0.99 · x` floors values into [0.01, 1], preserving order while avoiding zero-product collapse.

**How to read:** Higher score = better. Options with consistently good scores on all criteria rank higher than those with one weak spot.

**When to use:** When you want compensatory trade-offs; scores on all criteria must contribute (no ignoring weak dimensions).

---

### 7. SENSITIVITY_TEST (IRR Weight Sweep)

**What it is:** Repeatedly ranks all options while varying the IRR weight across a range.

**Range tested:** [0.05, 0.114, 0.179, 0.243, 0.307, 0.371, 0.436, 0.50] — 8 evenly-spaced points from near-zero to 0.50.

**Output:** A table and utility curve for each weight value, showing how the ranking responds to IRR emphasis.

**How to read:** If the top 5 vendors remain stable across all IRR weights, the ranking is robust. If they shift dramatically, the result is sensitive to IRR weighting — you should validate your IRR data and weight.

**When to use:** Robustness check; verify the final ranking is not brittle to weight assumptions.

---

### 8. COMPARE (All Methods Side-by-Side)

**What it is:** Runs all 6 core methods (MANUAL, AHP, LOSS, TOPSIS, VIKOR, WPM) on the same data, normalizes each method's scores independently, and aggregates via **Borda consensus ranking**.

**Borda consensus:** Each method ranks options 1 to N. Points awarded = `N − rank`. Options with the highest total points across all methods are the consensus favorites.

**Output:**
1. **Wide table:** Vendor name, S.no, and for each method: normalized score, method rank, plus a **Consensus (Borda)** column
2. **Grouped bar chart:** Top 15 options by consensus, showing normalized scores for all methods

**How to read:** The Borda column (far right of the table) is the overall ranking. The bar chart shows which vendors are broad consensus picks (tall bars across all methods) vs. method-specific favorites.

**When to use:** Decision review; see which vendors are robustly favored across diverse methods, and which are method-sensitive outliers.

---

## Sidebar Controls

### Method selector
Dropdown to choose which method to run. Some methods ignore the weight sliders (AHP, LOSS).

### Cost criteria toggle
**Label:** "Invert cost criteria — Reputational, Market & Business, Regulatory risk (recommended)"

When **on** (default): the 3 risk attributes are inverted via `1 − x` so that higher is always better.
When **off**: risk attributes are treated as benefits (higher raw score = better), which is usually wrong.

### Weight sliders
Nine sliders (one per attribute), initialized to `MANUAL_WEIGHTS`. Used by MANUAL, TOPSIS, VIKOR, WPM, and SENSITIVITY.

When the sum of sliders ≠ 1.0, they're automatically normalized and a note shows the effective percentages (e.g., "IRR=24%, Strategic fit=12%").

**AHP and LOSS ignore these sliders** — they use weights from the pairwise matrix.

---

## AHP Pairwise-Comparison Matrix

The matrix is fixed in `scalequest/config.py` and represents the relative importance of each criterion:

```
         IRR  SF  TF  UR  RR  MBR  SC  RgR  MF
IRR       1   3   5   3   7    9   5    7   3
SF       1/3  1   3   3   7    7   5    7   3
TF       1/5 1/3  1  1/3  3    3   3    5   1
UR       1/3 1/3  3   1   5    7   5    7   3
RR       1/7 1/7 1/3 1/5  1    3   3    5   1
MBR      1/9 1/7 1/3 1/7 1/3  1  1/3   1  1/3
SC       1/5 1/5 1/3 1/5 1/3  3   1    3  1/3
RgR      1/7 1/7 1/5 1/7 1/5  1  1/3   1  1/5
MF       1/3 1/3  1  1/3  1    3   3    5   1
```

**Derived weights:** [IRR: 0.25, SF: 0.15, TF: 0.07, UR: 0.14, RR: 0.08, MBR: 0.04, SC: 0.07, RgR: 0.05, MF: 0.15]

**Consistency ratio (CR):** 0.064 (acceptable; ≤ 0.10 is the standard threshold).

The matrix shows that IRR and Market factors are the strongest criteria, while Market & Business risk and Scalability are relatively weak.

---

## Project Structure

```
ScaleQuest-MAUT_Application/
├── app.py                      # Streamlit entry point; UI + method dispatch
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── scalequest/                 # Python package
│   ├── __init__.py
│   ├── config.py               # Paths, AHP matrix, weights, constants
│   ├── preprocessing.py        # Data loading, normalization, caching
│   ├── scoring.py              # All scoring engines (pure; no config import)
│   ├── charts.py               # Altair charts (no matplotlib)
│   ├── methods.py              # The 8 methods; orchestrate scoring + UI
│
├── data/
│   ├── Consolidated data.xlsx  # Sample dataset (~140 vendors × 9 attributes)
│   └── Consolidated data_normalized.csv  # Reference normalized snapshot
│
├── assets/diagrams/            # Concept + "working" diagrams (*.drawio.png)
│   ├── maut_1-MAUT_1.drawio.png
│   ├── maut_1-MAUT_1-WORKING.drawio.png
│   ├── maut_1-AHP.drawio.png
│   ├── maut_1-AHP-WORKING.drawio.png
│   ├── maut_1-LOSS.drawio.png
│   ├── maut_1-LOSS-WORKING.drawio.png
│   ├── maut_1-TOPSIS.drawio.png
│   └── maut_1-TOPSIS-WORKING.drawio.png
│
└── notebooks/                  # Jupyter prototypes for each method
    ├── MAUT_1.ipynb
    ├── MAUT_AHP.ipynb
    ├── MAUT_LOSS.ipynb
    ├── MAUT_TOPSIS.ipynb
    └── main_maut.ipynb
```

---

## Expected Excel Data Format

The input `.xlsx` file must have:

1. **Header row** (row 1) — skipped automatically
2. **Columns:**
   - Column A: Index (integer, 1 to N) — dropped
   - Columns B–D: `Company` (string), `Vendor` (string), optional notes
   - Columns E–M: The 9 attribute scores (numeric or coercible to numeric)
   - Column N+: Extra columns (preserved as-is)
3. **Trailing rows:** 2 summary/footer rows at the bottom — dropped automatically
4. **Total:** minimum 14 rows (header + 2 footer + 12 data rows)

**Example structure:**
```
Row 1: [Header] [Desc] [Company] [Vendor] [IRR] [SF] [TF] [UR] [RR] [MBR] [SC] [RgR] [MF]
Row 2: 1 note Company A Vendor A 80 75 85 70 60 65 75 70 80
Row 3: 2 note Company B Vendor B 70 80 75 75 70 75 80 75 75
...
Row N: N note Company N Vendor N ...
Row N+1: (summary/footer)
Row N+2: (summary/footer)
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements include: Streamlit, pandas, numpy, scikit-learn, openpyxl, altair.

### 2. Run the app

From the repository root:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 3. Try it out

1. Upload `data/Consolidated data.xlsx`
2. Select "MANUAL" from the method dropdown
3. Observe the ranked table and utility curve
4. Switch methods and compare results
5. Try the weight sliders (ignored by AHP/LOSS)
6. Run "COMPARE" to see all methods side-by-side

---

## Notebooks

Jupyter notebooks in `notebooks/` explore each method in isolation:

- **MAUT_1.ipynb** — Weighted sum (MANUAL) on a small hand-crafted example
- **MAUT_AHP.ipynb** — AHP pairwise-comparison matrix, weights, and consistency ratio
- **MAUT_LOSS.ipynb** — LOSS exponential and quadratic variants
- **MAUT_TOPSIS.ipynb** — TOPSIS ideal-point calculation and closeness formula
- **main_maut.ipynb** — Integrated workflow combining all methods on the full dataset

These are for reference and exploration; the app is the primary interface.

---

## Key Design Decisions

### 1. AHP uses geometric-mean weights (not eigenvector)
The geometric-mean method `w_j = (∏ a_{j,k})^(1/n)` is faster and scales better than eigenvalue decomposition, and is a standard approximation.

### 2. Cache keyed on file bytes, not UploadedFile object
Streamlit's `UploadedFile` object is a new instance on every rerun, so it's unhashable as a cache key. Instead, we cache on `file.getvalue()` (the bytes), which is content-addressed. This makes slider drags and method switches reuse the same cached frame.

### 3. WPM floor fix: `0.01 + 0.99 · x`
Min-Max normalization creates structural zeros (each column's min = 0). One zero factor collapses the product. We floor into `[0.01, 1]` to eliminate zeros while preserving order.

### 4. VIKOR stores `overall_score = −Q`
TOPSIS, MANUAL, LOSS, WPM all use `score = higher is better`. VIKOR's Q index is inverted (`lower is better`), so we store `−Q` so the shared `rank_and_select` function (DESC sort) works uniformly.

### 5. Cost inversion: `1 − x` on normalized data
Risk criteria are costs (minimize). After Min-Max normalization to `[0, 1]`, we invert via `1 − x` so downstream math treats all attributes as benefits (`higher is better`).

### 6. No intermediate columns in scoring
Older code used `data.filter(like="_weighted")` to sum, which accumulated stale columns on frame reuse. Current code computes scores directly: `sum(data[a] * w for a, w in weights.items())`.

### 7. Altair for charts (no matplotlib)
matplotlib requires explicit figure cleanup and can leak memory in long-running Streamlit sessions. Altair generates JSON-backed interactive charts with no global state.

### 8. Separate `scoring.py` never imports `config`
`config.py` bottom-imports from `scoring` (to precompute AHP weights). This one-way dependency prevents circular imports. `scoring.py` is pure (numpy + pandas only) and receives all config values as arguments.

---

## Troubleshooting

### "Non-numeric values found in:" warning
Some cells in the attribute columns contain text, dates, or other non-numeric data. `pd.to_numeric(..., errors="coerce")` converts them to NaN, which are then scored as 0 (after Min-Max). Check your data for typos or formatting issues.

### "All options scored identically under this method"
All options ended up with the same score. Possible causes:
- All attribute values are identical (degenerate data)
- All weights are zero (check sliders)
- Cost toggle is off and risk attributes are very weak

Try a different method or inspect the input data.

### Rankings differ between MANUAL and AHP
MANUAL uses the weight sliders; AHP uses the fixed pairwise matrix. Their weights are different, so rankings diverge. This is expected.

### TOPSIS closeness is 0.5 for all options
All options are equidistant from the ideal. This happens when options cluster uniformly or attribute ranges overlap symmetrically. Run SENSITIVITY_TEST to check robustness.

---

## License

ScaleQuest is released under the **MIT License**. See [LICENSE](LICENSE) for full details.

**TL;DR:** You're free to use, modify, and distribute this software commercially and privately, with no warranty. Just include a copy of the license.

---

## Citation

If you use ScaleQuest in academic work, please cite as:

```
Sojitra-Nency. (2026). ScaleQuest: MCDA Ranking Tool for Mining R&D.
Retrieved from https://github.com/sojitra-nency/ScaleQuest-MAUT_Application
```

---

## Authors & Contributors

- **Sojitra-Nency** — primary author and maintainer
- Built with Streamlit, pandas, numpy, scikit-learn, and Altair

---

## Questions & Support

For issues, feature requests, bug reports, or questions:
- Open an issue on [GitHub](https://github.com/sojitra-nency/ScaleQuest-MAUT_Application/issues)
- Check existing documentation in this README and the `/notebooks` folder
- Review the design decisions section for understanding how the app works
