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

## Artefacts (for group reuse)

The following CSVs are committed to the repo so the group can drop them into the week-2/6/8 pipelines without re-running this notebook:

- `features_table.csv` — schema + counts for the week-4 feature set; use the column list as the contract for other week cutoffs
- `pipeline_comparison.csv` — full KPI baseline (P1 vs P2) for week 4; group adds week-2/6/8 columns alongside
- `kpi_results.csv` — per-pipeline metrics in long format for joining with other weeks
- `appx_fairness_audit_by_imd.csv` — per-IMD-band FPR/FNR/Recall/Precision (test set, threshold 0.35, full model)
- `appx_ablation_no_imd.csv` — imd_band ablation deltas (AUC/F1/Recall with vs without imd_band)
- `appx_fairness_audit_no_imd.csv` — per-IMD-band audit on the ablated model

All CSVs are produced by `OULAD_Individual_2434427.ipynb`. To regenerate, place `studentInfo.csv`, `studentVle.csv`, and `vle.csv` in the root directory and run the notebook top to bottom — outputs land in the root.

## Results

XGBoost outperforms Logistic Regression on AUC and log loss (see `pipeline_comparison.csv` for full KPI table). Both confusion matrices and ROC curves are in the notebook.
