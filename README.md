# PSA-i Proteomics Prostate Cancer Study

## Overview
The manuscript associated with this repository is currently under review. This repository will be updated with the associated code base and abstract after publication.

This folder now includes a public-facing analysis script designed for external review. Further details and changes may be made after publication and review.

- `Proteomics_Main.py`

The script reproduces the core workflow from the internal master analysis while improving readability and making path usage anonymized.

## Scientific Purpose
The script evaluates prostate cancer risk prediction models over fixed time horizons (5 years and 10 years) using:

1. Discrimination metrics
   - AUC
   - AUPRC
2. Biopsy simulation
   - Top 5% model-risk threshold as a proxy recommendation rule
3. Time-to-event analysis
   - Kaplan-Meier curves
   - Log-rank testing
4. Explainability summaries
   - SHAP-ranked feature importance for XGBoost models

## Required Input Files
Place these files in one input directory:

1. `baseline_proteomics.csv`
2. `showcase.csv`
3. `prostate_ca.csv`

## Minimum Required Columns
The script expects these key columns to exist.

### In baseline_proteomics.csv
- `ID`
- `KLK3.Prostate.specific.antigen`
- proteomics feature columns

### In showcase.csv
- `ID`
- `BaC_Sex_x`
- at least one age-like column (contains "age" in its column name)

### In prostate_ca.csv
- `eid`
- `T` (follow-up time in days)
- `E` (event indicator)

## Path Anonymization (Fill-In-Blanks)
The script intentionally uses placeholders. Provide anonymized real paths at runtime.

Example:

```bash
python Proteomics_Main.py \
  --input-dir /path/to/ANONYMIZED_INPUT_DIR \
  --output-dir /path/to/ANONYMIZED_OUTPUT_DIR
```

If placeholders are not replaced, the script will stop with a clear error.

## Run Instructions
1. Create or activate an environment with required packages.
2. Confirm all required input files and columns are present.
3. Run the script with anonymized input/output paths.
4. Generated CSV and PDF outputs in the output directory.

## Python Dependencies
Install these packages if needed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```

## Main Output Files
The script writes tables and figures, including:

- `results_AUC.csv`
- `results_AUPRC.csv`
- `biopsy_results.csv`
- `km_results.csv`
- `klk3_medians.csv`
- `best_hyperparameters.csv`
- `feature_importances.csv`
- `km_plot_base.csv`
- `km_plot_scores.csv`
- `km_quintile_summary.csv`
- `km_quintile_labels.csv`
- `klk3_age_testset.csv`
- plot PDFs (`Public_*.pdf`)

## Notes for Reviewers
- Cohort inclusion is horizon-specific:
  - positive if event occurred by horizon
  - negative if event-free beyond horizon
- The train/test split is fixed once globally, then intersected with each horizon-specific cohort.
- Proteomics features are pre-selected with SHAP ranking on training data only.
- Model set:
  - KLK3-only logistic regression
  - Proteomics-only XGBoost
  - KLK3+Proteomics XGBoost

## Reproducibility
Key reproducibility artifacts are saved:

- per-horizon model scores (`km_plot_scores.csv`)
- per-horizon base KM inputs (`km_plot_base.csv`)
- quintile assignments (`km_quintile_labels.csv`)
- selected hyperparameters (`best_hyperparameters.csv`)
