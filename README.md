# ScaleQuest

Welcome to the **ScaleQuest-MAUT** (Multi-Attribute Utility Theory) application! This repository
contains a decision-making tool that ranks options across multiple weighted criteria using several
MAUT / multi-criteria decision-analysis techniques. The app is hosted on Streamlit at
[https://maut-1234.streamlit.app/](https://maut-1234.streamlit.app/).

You upload an Excel dataset, choose an approach in the sidebar, and ScaleQuest normalizes the data,
scores every option, ranks them, and plots a utility curve.

## Decision methods

- **MANUAL** — weighted-sum utility using fixed, hand-set attribute weights.
- **AHP** (Analytic Hierarchy Process) — weights derived from a pairwise-comparison matrix via the
  geometric-mean method, then a weighted-sum utility.
- **LOSS** (Level of Service Satisfaction) — AHP-derived weights applied through an exponential
  utility function.
- **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) — ranks options by
  their relative closeness to the ideal-best and ideal-worst solutions.
- **SENSITIVITY_TEST** — sweeps the IRR weight to show how the ranking responds.

## Project structure

```
app.py                 # Streamlit entry point (UI + method dispatch)
scalequest/            # Application package
  config.py            # Paths, diagram pairs, and all weights / the AHP matrix
  preprocessing.py     # Shared data loading + Min-Max normalization
  scoring.py           # Scoring engines + ranking helper
  plotting.py          # Utility-curve plot
  methods.py           # The five MAUT methods
data/                  # Consolidated data.xlsx (sample) + normalized CSV
assets/diagrams/       # Method / workflow diagrams (*.drawio.png)
notebooks/             # Prototype notebooks for each method
requirements.txt
```

## Data

- `data/Consolidated data.xlsx` — sample dataset you can upload to try the app.
- `data/Consolidated data_normalized.csv` — a pre-normalized snapshot for reference.

## Notebooks

Prototype analyses for each method live in `notebooks/`: `MAUT_1.ipynb`, `MAUT_AHP.ipynb`,
`MAUT_LOSS.ipynb`, `MAUT_TOPSIS.ipynb`, and `main_maut.ipynb`.

## Visualization

Workflow diagrams (concept + "working" pairs) are in `assets/diagrams/`, e.g.
`maut_1-AHP.drawio.png` / `maut_1-AHP-WORKING.drawio.png`, and likewise for `LOSS`, `TOPSIS`, and
`MAUT_1`. They are shown alongside each method in the app.

## Getting started

1. Install the dependencies: `pip install -r requirements.txt`.
2. Launch the app from the repository root: `streamlit run app.py`.
3. Open `http://localhost:8501`, upload an `.xlsx` dataset (e.g. `data/Consolidated data.xlsx`),
   and pick an approach.
