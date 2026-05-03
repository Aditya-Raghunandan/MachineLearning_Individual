# OULAD Week-4 Fail/Pass Prediction

Early at-risk prediction for Open University students using only their first 4 weeks of data.

## What this does

Takes VLE click data up to day 28 and predicts whether a student will pass or fail. Compares two pipelines — Logistic Regression vs XGBoost with tuned hyperparameters.

## Data

[OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) — download and place these in the root directory:
- `studentInfo.csv`
- `studentVle.csv`
- `vle.csv`

## Setup

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Run

Open `OULAD_Individual_2434427.ipynb` and run all cells top to bottom.

## Key design decisions

- Cutoff enforced on the raw click table *before* aggregation (no leakage)
- `GroupShuffleSplit` on `id_student` so the same student can't appear in both train and test
- Missing `imd_band` kept as its own category instead of imputed

## Results

XGBoost outperforms Logistic Regression on AUC and log loss. Both confusion matrices and ROC curves are in the notebook.

## Artefacts
Artefacts for group reuse: features_table.csv is the week-4 feature schema, drop into the group pipeline as the contract for week-2/6/8 builders. pipeline_comparison.csv is the KPI baseline.
