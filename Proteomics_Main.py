#!/usr/bin/env python3
"""
Public-facing prostate cancer risk analysis script.

Purpose
-------
This script provides an overview of our implementation of a multi-model prostate
cancer risk analysis pipeline that combines:
1. AUC-based discrimination analysis (with subgroup stratification)
2. AUPRC-based discrimination analysis (with subgroup stratification)
3. Biopsy recommendation simulation (top 5% risk threshold)
4. Kaplan-Meier survival analysis and log-rank testing
5. Model explainability summaries (SHAP-based feature ranking)

This version is designed for external review. It keeps scientific logic clear,
uses descriptive variable names, and avoids embedding environment-specific paths.

Privacy / Anonymity
-------------------
Before running, fill in anonymized input and output paths using command line
arguments or by editing the defaults in parse_args().

Expected input files
--------------------
- baseline_proteomics.csv
- showcase.csv
- prostate_ca.csv

Expected major output files
---------------------------
- results_AUC.csv
- results_AUPRC.csv
- biopsy_results.csv
- km_results.csv
- feature_importances.csv
- best_hyperparameters.csv
- km_quintile_summary.csv
- km_quintile_labels.csv
- multiple plot PDFs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None


MODEL_ORDER = ["klk3_only", "proteomics_only", "klk3_plus_proteomics"]
MODEL_LABELS = {
    "klk3_only": "KLK3",
    "proteomics_only": "Proteomics",
    "klk3_plus_proteomics": "KLK3+Proteomics",
}
MODEL_COLORS = {
    "klk3_only": "#1f77b4",
    "proteomics_only": "#ff7f0e",
    "klk3_plus_proteomics": "#2ca02c",
}


@dataclass(frozen=True)
class ColumnNames:
    participant_id: str = "ID"
    outcome_id: str = "eid"
    sex: str = "BaC_Sex_x"
    male_value: str = "Male"
    followup_days: str = "T"
    event_flag: str = "E"
    klk3_marker: str = "KLK3.Prostate.specific.antigen"
    preferred_age_column: str = "BaC_Age_x"


@dataclass
class AnalysisConfig:
    input_dir: str
    output_dir: str
    input_filenames: Dict[str, str] = field(
        default_factory=lambda: {
            "proteomics": "baseline_proteomics.csv",
            "demographics": "showcase.csv",
            "outcomes": "prostate_ca.csv",
        }
    )
    horizons: Tuple[Tuple[str, int, int], ...] = (
        ("5y", 0, 5 * 365),
        ("10y", 0, 10 * 365),
    )
    random_seed: int = 42
    bootstraps: int = 1000
    top_shap_features: int = 15
    min_group_n: int = 8
    min_horizon_n: int = 50
    test_size: float = 0.30
    cv_folds: int = 5
    cv_jobs: int = 4
    xgb_random_search_iters: int = 200
    age_cutoff_years: int = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Public-facing prostate risk analysis with AUC/AUPRC/biopsy/KM outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="<FILL_IN_INPUT_DIR>",
        help="Folder containing baseline_proteomics.csv, showcase.csv, and prostate_ca.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="<FILL_IN_OUTPUT_DIR>",
        help="Folder where all output CSV and PDF files will be written.",
    )
    parser.add_argument("--bootstraps", type=int, default=1000, help="Bootstrap resamples for confidence intervals.")
    parser.add_argument("--xgb-random-search-iters", type=int, default=200, help="RandomizedSearchCV iterations for XGBoost models.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--cv-jobs", type=int, default=4, help="Parallel jobs for RandomizedSearchCV.")
    parser.add_argument("--top-shap-features", type=int, default=15, help="Number of SHAP-ranked proteomics features retained.")
    return parser.parse_args()


def assert_filled_path(path_value: str, argument_name: str) -> None:
    if "<FILL_IN" in path_value:
        raise ValueError(
            f"{argument_name} is still a placeholder ({path_value}). "
            "Replace it with an anonymized real path before running."
        )


def build_config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bootstraps=args.bootstraps,
        xgb_random_search_iters=args.xgb_random_search_iters,
        cv_folds=args.cv_folds,
        cv_jobs=args.cv_jobs,
        top_shap_features=args.top_shap_features,
    )


def validate_and_prepare_paths(config: AnalysisConfig) -> None:
    assert_filled_path(config.input_dir, "--input-dir")
    assert_filled_path(config.output_dir, "--output-dir")

    if not os.path.isdir(config.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")

    os.makedirs(config.output_dir, exist_ok=True)


def build_input_path(config: AnalysisConfig, key: str) -> str:
    filename = config.input_filenames[key]
    path = os.path.join(config.input_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def safe_filename(text: str) -> str:
    clean = str(text).strip().replace(" ", "_").replace("/", "-")
    clean = re.sub(r"[^0-9A-Za-z_\-.]", "", clean)
    return clean


def save_figure(fig: plt.Figure, output_dir: str, filename_base: str, ext: str = "pdf", dpi: int = 150) -> None:
    output_path = os.path.join(output_dir, f"{safe_filename(filename_base)}.{ext}")
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def detect_non_numeric_columns(df: pd.DataFrame) -> List[str]:
    drop_columns: List[str] = []
    for column in df.columns:
        converted = pd.to_numeric(df[column].dropna(), errors="coerce")
        if not converted.notna().all():
            drop_columns.append(column)
    return drop_columns


def load_input_tables(config: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proteomics_path = build_input_path(config, "proteomics")
    demographics_path = build_input_path(config, "demographics")
    outcomes_path = build_input_path(config, "outcomes")

    print("Loading input files...")
    proteomics_df = pd.read_csv(proteomics_path)
    demographics_df = pd.read_csv(demographics_path)
    outcomes_df = pd.read_csv(outcomes_path)
    print(f"Proteomics rows: {len(proteomics_df)}")
    print(f"Demographics rows: {len(demographics_df)}")
    print(f"Outcome rows: {len(outcomes_df)}")
    return proteomics_df, demographics_df, outcomes_df


def merge_and_clean_tables(
    proteomics_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    columns: ColumnNames,
) -> pd.DataFrame:
    for required_col in [columns.participant_id, columns.klk3_marker]:
        if required_col not in proteomics_df.columns:
            raise KeyError(f"Missing required proteomics column: {required_col}")

    for required_col in [columns.participant_id, columns.sex]:
        if required_col not in demographics_df.columns:
            raise KeyError(f"Missing required demographics column: {required_col}")

    for required_col in [columns.outcome_id, columns.followup_days, columns.event_flag]:
        if required_col not in outcomes_df.columns:
            raise KeyError(f"Missing required outcomes column: {required_col}")

    demographics_clean = demographics_df.copy()
    if "TEU_BaC_AgeAtRec" in demographics_clean.columns:
        demographics_clean = demographics_clean.drop(columns=["TEU_BaC_AgeAtRec"])

    merged_features_df = pd.merge(
        proteomics_df,
        demographics_clean,
        on=columns.participant_id,
        how="inner",
    )

    merged_features_df = merged_features_df[merged_features_df[columns.sex] == columns.male_value].copy()

    # Keep only fully numeric fields before joining to outcomes.
    non_numeric_columns = detect_non_numeric_columns(merged_features_df)
    merged_features_df = merged_features_df.drop(columns=non_numeric_columns)

    cohort_df = pd.merge(
        merged_features_df,
        outcomes_df,
        left_on=columns.participant_id,
        right_on=columns.outcome_id,
        how="inner",
    )

    print(f"Merged analysis cohort rows: {len(cohort_df)}")
    return cohort_df


def km_step(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    valid = (~np.isnan(times)) & (~np.isnan(events))
    times = times[valid]
    events = events[valid]

    if len(times) == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times)
    n_at_risk = len(times)
    survival = 1.0
    x_values = [0.0]
    y_values = [1.0]

    idx_start = 0
    for current_time in unique_times:
        idx_end = np.searchsorted(times, current_time, side="right")
        event_count = int(events[idx_start:idx_end].sum())
        if n_at_risk > 0 and event_count > 0:
            survival *= 1.0 - event_count / n_at_risk

        x_values.append(float(current_time))
        y_values.append(float(survival))

        n_at_risk -= idx_end - idx_start
        idx_start = idx_end

    return np.array(x_values), np.array(y_values)


def two_sample_logrank_pvalue(
    times_a: np.ndarray,
    events_a: np.ndarray,
    times_b: np.ndarray,
    events_b: np.ndarray,
) -> Tuple[float, float]:
    if chi2 is None:
        return np.nan, np.nan

    times_a = np.asarray(times_a, dtype=float)
    events_a = np.asarray(events_a, dtype=int)
    times_b = np.asarray(times_b, dtype=float)
    events_b = np.asarray(events_b, dtype=int)

    valid_a = (~np.isnan(times_a)) & (~np.isnan(events_a))
    valid_b = (~np.isnan(times_b)) & (~np.isnan(events_b))
    times_a, events_a = times_a[valid_a], events_a[valid_a]
    times_b, events_b = times_b[valid_b], events_b[valid_b]

    if len(times_a) == 0 or len(times_b) == 0:
        return np.nan, np.nan

    pooled_event_times = np.unique(np.concatenate([times_a[events_a == 1], times_b[events_b == 1]]))
    if len(pooled_event_times) == 0:
        return np.nan, np.nan

    observed_minus_expected = 0.0
    variance = 0.0

    for current_time in pooled_event_times:
        n_a = int(np.sum(times_a >= current_time))
        n_b = int(np.sum(times_b >= current_time))
        n_total = n_a + n_b
        if n_total <= 1:
            continue

        d_a = int(np.sum((times_a == current_time) & (events_a == 1)))
        d_b = int(np.sum((times_b == current_time) & (events_b == 1)))
        d_total = d_a + d_b
        if d_total == 0:
            continue

        expected_a = d_total * (n_a / n_total)
        var = 0.0
        if n_total > 1:
            var = (n_a * n_b * d_total * (n_total - d_total)) / (n_total * n_total * (n_total - 1))

        observed_minus_expected += d_a - expected_a
        variance += var

    if variance <= 0:
        return np.nan, np.nan

    chi2_stat = (observed_minus_expected ** 2) / variance
    pvalue = float(chi2.sf(chi2_stat, df=1))
    return pvalue, float(chi2_stat)


def k_sample_logrank_pvalue(times_by_group: List[np.ndarray], events_by_group: List[np.ndarray]) -> Tuple[float, float, int]:
    if chi2 is None:
        return np.nan, np.nan, 0

    cleaned_times: List[np.ndarray] = []
    cleaned_events: List[np.ndarray] = []

    for times, events in zip(times_by_group, events_by_group):
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)
        valid = (~np.isnan(times)) & (~np.isnan(events))
        times = times[valid]
        events = events[valid]
        if len(times) > 0:
            cleaned_times.append(times)
            cleaned_events.append(events)

    k_groups = len(cleaned_times)
    if k_groups < 2:
        return np.nan, np.nan, 0

    event_times = np.unique(np.concatenate([t[e == 1] for t, e in zip(cleaned_times, cleaned_events)]))
    if len(event_times) == 0:
        return np.nan, np.nan, k_groups

    observed = np.zeros(k_groups, dtype=float)
    expected = np.zeros(k_groups, dtype=float)
    covariance = np.zeros((k_groups, k_groups), dtype=float)

    for current_time in event_times:
        n_i = np.array([np.sum(group_times >= current_time) for group_times in cleaned_times], dtype=float)
        d_i = np.array(
            [np.sum((group_times == current_time) & (group_events == 1)) for group_times, group_events in zip(cleaned_times, cleaned_events)],
            dtype=float,
        )

        n_total = n_i.sum()
        d_total = d_i.sum()
        if n_total <= 1 or d_total == 0:
            continue

        expected_i = d_total * (n_i / n_total)
        observed += d_i
        expected += expected_i

        factor = (d_total * (n_total - d_total)) / (n_total * n_total * (n_total - 1))
        for i in range(k_groups):
            for j in range(k_groups):
                if i == j:
                    covariance[i, j] += factor * n_i[i] * (n_total - n_i[i])
                else:
                    covariance[i, j] += -factor * n_i[i] * n_i[j]

    reduced_observed = observed[:-1]
    reduced_expected = expected[:-1]
    reduced_cov = covariance[:-1, :-1]

    try:
        chi2_stat = float((reduced_observed - reduced_expected).T @ np.linalg.pinv(reduced_cov) @ (reduced_observed - reduced_expected))
        pvalue = float(chi2.sf(chi2_stat, df=k_groups - 1))
    except Exception:
        return np.nan, np.nan, k_groups

    return pvalue, chi2_stat, k_groups


def bootstrap_metric_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstraps: int,
    rng_seed: int,
) -> np.ndarray:
    if y_score is None:
        return np.array([])

    rng = np.random.default_rng(rng_seed)
    n_samples = len(y_true)
    values: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_samples, n_samples)
        y_boot = y_true[boot_idx]
        score_boot = y_score[boot_idx]

        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue

        try:
            values.append(float(metric_fn(y_boot, score_boot)))
        except Exception:
            continue

    return np.asarray(values, dtype=float)


def xgb_mean_abs_shap(model: XGBClassifier, matrix: np.ndarray, feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(matrix, feature_names=list(feature_names))
        shap_values = booster.predict(dmatrix, pred_contribs=True, validate_features=False)

        if shap_values.ndim != 2:
            return []

        if shap_values.shape[1] == len(feature_names) + 1:
            shap_values = shap_values[:, :-1]

        if shap_values.shape[1] != len(feature_names):
            return []

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        ranked = list(zip(feature_names, mean_abs_shap.tolist()))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
    except Exception:
        return []


def find_age_columns(cohort_df: pd.DataFrame, columns: ColumnNames) -> List[str]:
    return [
        column
        for column in cohort_df.columns
        if "age" in column.lower() and column not in [columns.followup_days, columns.event_flag]
    ]


def choose_age_column(age_candidates: List[str], columns: ColumnNames) -> str:
    if len(age_candidates) == 0:
        raise ValueError("No age-like column was found in the merged cohort.")
    if columns.preferred_age_column in age_candidates:
        return columns.preferred_age_column
    return age_candidates[0]


def prepare_base_population(cohort_df: pd.DataFrame, columns: ColumnNames) -> pd.DataFrame:
    base_df = cohort_df.dropna(subset=[columns.followup_days, columns.event_flag]).copy()
    base_df[columns.followup_days] = pd.to_numeric(base_df[columns.followup_days], errors="coerce")
    base_df[columns.event_flag] = pd.to_numeric(base_df[columns.event_flag], errors="coerce").astype(int)
    base_df = base_df.loc[base_df[columns.followup_days] >= 0].reset_index(drop=True)

    if len(base_df) == 0:
        raise ValueError("Base population is empty after filtering for non-missing follow-up and event labels.")

    return base_df


def default_xgb_static_params(seed: int, n_jobs: int) -> Dict[str, Any]:
    return {
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.10,
        "subsample": 1.0,
        "colsample_bytree": 0.5,
        "reg_lambda": 5,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }


def default_xgb_random_search_grid() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150, 200, 300, 400, 450],
        "max_depth": [2, 3, 4, 6, 8, 10, 12],
        "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20],
        "gamma": [0, 0.5, 1, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.3, 0.5, 0.6, 0.7, 0.8, 1.0],
        "reg_lambda": [0, 0.5, 1, 3, 5, 10, 11],
        "reg_alpha": [0, 0.1, 0.5, 1, 3],
        "scale_pos_weight": [1, 2, 4, 8, 20, 40],
    }


def build_subgroup_masks(
    klk3_values: np.ndarray,
    age_values: np.ndarray,
    klk3_median: float,
    age_cutoff: int,
) -> Dict[str, Dict[str, Any]]:
    age_lt_mask = age_values < age_cutoff
    age_ge_mask = age_values >= age_cutoff

    subgroups = {
        "klk3_le_median": {
            "mask": klk3_values <= klk3_median,
            "label": f"KLK3 <= {klk3_median:.3f}",
            "category": "overall",
        },
        "klk3_gt_median": {
            "mask": klk3_values > klk3_median,
            "label": f"KLK3 > {klk3_median:.3f}",
            "category": "overall",
        },
        "age_lt_cutoff_klk3_le_median": {
            "mask": age_lt_mask & (klk3_values <= klk3_median),
            "label": f"Age<{age_cutoff}, KLK3 <= {klk3_median:.3f}",
            "category": "age_stratified",
            "age_group": f"Age<{age_cutoff}",
        },
        "age_lt_cutoff_klk3_gt_median": {
            "mask": age_lt_mask & (klk3_values > klk3_median),
            "label": f"Age<{age_cutoff}, KLK3 > {klk3_median:.3f}",
            "category": "age_stratified",
            "age_group": f"Age<{age_cutoff}",
        },
        "age_ge_cutoff_klk3_le_median": {
            "mask": age_ge_mask & (klk3_values <= klk3_median),
            "label": f"Age>={age_cutoff}, KLK3 <= {klk3_median:.3f}",
            "category": "age_stratified",
            "age_group": f"Age>={age_cutoff}",
        },
        "age_ge_cutoff_klk3_gt_median": {
            "mask": age_ge_mask & (klk3_values > klk3_median),
            "label": f"Age>={age_cutoff}, KLK3 > {klk3_median:.3f}",
            "category": "age_stratified",
            "age_group": f"Age>={age_cutoff}",
        },
    }
    return subgroups


def collect_metric_rows(
    horizon_name: str,
    subgroup_masks: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    metric_fn,
    metric_name: str,
    n_bootstraps: int,
    rng_seed: int,
    min_group_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for subgroup_id, subgroup_info in subgroup_masks.items():
        subgroup_mask = subgroup_info["mask"]
        if subgroup_mask.sum() < min_group_n:
            continue

        y_group = y_true[subgroup_mask]
        if y_group.sum() < 1 or len(np.unique(y_group)) < 2:
            continue

        for model_name in MODEL_ORDER:
            scores_group = model_scores.get(model_name)
            if scores_group is None:
                continue

            score_slice = scores_group[subgroup_mask]
            observed_value = float(metric_fn(y_group, score_slice))
            boot_dist = bootstrap_metric_distribution(
                y_true=y_group,
                y_score=score_slice,
                metric_fn=metric_fn,
                n_bootstraps=n_bootstraps,
                rng_seed=rng_seed,
            )
            if len(boot_dist) > 0:
                ci_low, ci_high = np.percentile(boot_dist, [2.5, 97.5])
            else:
                ci_low, ci_high = np.nan, np.nan

            rows.append(
                {
                    "Horizon": horizon_name,
                    "SubgroupId": subgroup_id,
                    "SubgroupLabel": subgroup_info["label"],
                    "SubgroupCategory": subgroup_info["category"],
                    "Model": model_name,
                    metric_name: observed_value,
                    "CI_lo": ci_low,
                    "CI_hi": ci_high,
                    "N": int(subgroup_mask.sum()),
                    "Events": int(y_group.sum()),
                }
            )

    return rows


def youden_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    best_threshold: Optional[float] = None
    best_j_stat = -np.inf

    for threshold in np.unique(y_score):
        predicted_high = y_score >= threshold
        if predicted_high.sum() == 0 or predicted_high.sum() == len(predicted_high):
            continue

        tp = int(((predicted_high) & (y_true == 1)).sum())
        fp = int(((predicted_high) & (y_true == 0)).sum())
        fn = int(((~predicted_high) & (y_true == 1)).sum())
        tn = int(((~predicted_high) & (y_true == 0)).sum())

        if (tp + fn) == 0 or (tn + fp) == 0:
            continue

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        j_stat = sensitivity + specificity - 1.0

        if j_stat > best_j_stat:
            best_j_stat = j_stat
            best_threshold = float(threshold)

    return best_threshold


def build_model_matrices(
    base_df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    klk3_values_all: np.ndarray,
    selected_proteomics_features: List[str],
) -> Dict[str, np.ndarray]:
    x_train_proteomics = base_df.loc[train_indices, selected_proteomics_features].fillna(-1.0).to_numpy(dtype=np.float32)
    x_test_proteomics = base_df.loc[test_indices, selected_proteomics_features].fillna(-1.0).to_numpy(dtype=np.float32)

    x_train_klk3 = klk3_values_all[train_indices].reshape(-1, 1)
    x_test_klk3 = klk3_values_all[test_indices].reshape(-1, 1)

    x_train_combined = np.hstack([x_train_klk3, x_train_proteomics])
    x_test_combined = np.hstack([x_test_klk3, x_test_proteomics])

    return {
        "x_train_klk3": x_train_klk3,
        "x_test_klk3": x_test_klk3,
        "x_train_proteomics": x_train_proteomics,
        "x_test_proteomics": x_test_proteomics,
        "x_train_combined": x_train_combined,
        "x_test_combined": x_test_combined,
    }


def fit_models_for_horizon(
    x_mats: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    proteomics_features: List[str],
    config: AnalysisConfig,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, bool], Dict[Tuple[str, str], List[Tuple[str, float]]], List[Dict[str, Any]]]:
    del y_test

    xgb_static = default_xgb_static_params(seed=config.random_seed, n_jobs=config.cv_jobs)
    xgb_search_grid = default_xgb_random_search_grid()

    tuned_keys = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda"}
    xgb_base_params = {k: v for k, v in xgb_static.items() if k not in tuned_keys}

    cv_splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    model_scores: Dict[str, Optional[np.ndarray]] = {model_name: None for model_name in MODEL_ORDER}
    model_fit_ok: Dict[str, bool] = {model_name: False for model_name in MODEL_ORDER}
    feature_importances: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    hyperparameter_rows: List[Dict[str, Any]] = []

    # KLK3-only model (logistic regression)
    try:
        logistic_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)),
            ]
        )
        logistic_grid = {"logistic__C": [0.01, 0.1, 1.0, 10.0]}
        logistic_search = GridSearchCV(
            estimator=logistic_pipeline,
            param_grid=logistic_grid,
            scoring="roc_auc",
            cv=cv_splitter,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        logistic_search.fit(x_mats["x_train_klk3"], y_train)
        best_logistic = logistic_search.best_estimator_

        model_fit_ok["klk3_only"] = True
        model_scores["klk3_only"] = best_logistic.predict_proba(x_mats["x_test_klk3"])[:, 1]

        hyperparameter_rows.append(
            {
                "Model": "klk3_only",
                "Search": "GridSearchCV",
                "BestParams": json.dumps(logistic_search.best_params_, sort_keys=True),
                "BestCVScore": float(logistic_search.best_score_) if logistic_search.best_score_ is not None else np.nan,
            }
        )
    except Exception as err:
        print(f"KLK3-only model failed: {err}")

    # Proteomics-only model (XGBoost)
    try:
        xgb_for_search = XGBClassifier(**xgb_base_params)
        proteomics_search = RandomizedSearchCV(
            estimator=xgb_for_search,
            param_distributions=xgb_search_grid,
            n_iter=config.xgb_random_search_iters,
            scoring="roc_auc",
            cv=cv_splitter,
            random_state=config.random_seed,
            n_jobs=config.cv_jobs,
            verbose=0,
            refit=True,
        )
        proteomics_search.fit(x_mats["x_train_proteomics"], y_train)
        best_proteomics_model = proteomics_search.best_estimator_

        model_fit_ok["proteomics_only"] = True
        model_scores["proteomics_only"] = best_proteomics_model.predict_proba(x_mats["x_test_proteomics"])[:, 1]

        shap_ranked = xgb_mean_abs_shap(best_proteomics_model, x_mats["x_train_proteomics"], proteomics_features)
        if len(shap_ranked) == 0:
            shap_ranked = list(zip(proteomics_features, best_proteomics_model.feature_importances_.tolist()))
        feature_importances[("proteomics_only", "train")] = shap_ranked

        hyperparameter_rows.append(
            {
                "Model": "proteomics_only",
                "Search": "RandomizedSearchCV",
                "BestParams": json.dumps(proteomics_search.best_params_, sort_keys=True),
                "BestCVScore": float(proteomics_search.best_score_) if proteomics_search.best_score_ is not None else np.nan,
            }
        )
    except Exception as err:
        print(f"Proteomics-only model failed: {err}")

    # Combined KLK3+Proteomics model (XGBoost)
    try:
        xgb_for_search = XGBClassifier(**xgb_base_params)
        combined_search = RandomizedSearchCV(
            estimator=xgb_for_search,
            param_distributions=xgb_search_grid,
            n_iter=config.xgb_random_search_iters,
            scoring="roc_auc",
            cv=cv_splitter,
            random_state=config.random_seed + 1,
            n_jobs=config.cv_jobs,
            verbose=0,
            refit=True,
        )
        combined_search.fit(x_mats["x_train_combined"], y_train)
        best_combined_model = combined_search.best_estimator_

        model_fit_ok["klk3_plus_proteomics"] = True
        model_scores["klk3_plus_proteomics"] = best_combined_model.predict_proba(x_mats["x_test_combined"])[:, 1]

        combined_features = ["KLK3"] + proteomics_features
        shap_ranked = xgb_mean_abs_shap(best_combined_model, x_mats["x_train_combined"], combined_features)
        if len(shap_ranked) == 0:
            shap_ranked = list(zip(combined_features, best_combined_model.feature_importances_.tolist()))
        feature_importances[("klk3_plus_proteomics", "train")] = shap_ranked

        hyperparameter_rows.append(
            {
                "Model": "klk3_plus_proteomics",
                "Search": "RandomizedSearchCV",
                "BestParams": json.dumps(combined_search.best_params_, sort_keys=True),
                "BestCVScore": float(combined_search.best_score_) if combined_search.best_score_ is not None else np.nan,
            }
        )
    except Exception as err:
        print(f"KLK3+Proteomics model failed: {err}")

    return model_scores, model_fit_ok, feature_importances, hyperparameter_rows


def run_biopsy_simulation(
    horizon_name: str,
    y_test: np.ndarray,
    age_test: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    age_cutoff: int,
    min_group_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    strategy_labels = {
        "klk3_only": "KLK3 only",
        "proteomics_only": "Proteomics",
        "klk3_plus_proteomics": "KLK3+Proteomics",
    }

    biopsy_masks: Dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        scores = model_scores[model_name]
        threshold = np.percentile(scores, 95)
        predicted_biopsy = scores >= threshold
        biopsy_masks[model_name] = predicted_biopsy

        tp = int(((predicted_biopsy) & (y_test == 1)).sum())
        fp = int(((predicted_biopsy) & (y_test == 0)).sum())
        fn = int(((~predicted_biopsy) & (y_test == 1)).sum())
        tn = int(((~predicted_biopsy) & (y_test == 0)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append(
            {
                "Horizon": horizon_name,
                "AgeGroup": None,
                "Model": model_name,
                "Strategy": strategy_labels[model_name],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "PPV": ppv,
                "N_Biopsies": int(predicted_biopsy.sum()),
                "N_Total": int(len(y_test)),
            }
        )

    age_groups = {
        f"Age<{age_cutoff}": age_test < age_cutoff,
        f"Age>={age_cutoff}": age_test >= age_cutoff,
    }

    for age_group_label, age_mask in age_groups.items():
        if age_mask.sum() < min_group_n:
            continue

        y_age = y_test[age_mask]
        for model_name in MODEL_ORDER:
            predicted_age = biopsy_masks[model_name][age_mask]

            tp = int(((predicted_age) & (y_age == 1)).sum())
            fp = int(((predicted_age) & (y_age == 0)).sum())
            fn = int(((~predicted_age) & (y_age == 1)).sum())
            tn = int(((~predicted_age) & (y_age == 0)).sum())

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            rows.append(
                {
                    "Horizon": horizon_name,
                    "AgeGroup": age_group_label,
                    "Model": model_name,
                    "Strategy": strategy_labels[model_name],
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "PPV": ppv,
                    "N_Biopsies": int(predicted_age.sum()),
                    "N_Total": int(len(y_age)),
                }
            )

    return rows


def run_horizon_analysis(
    horizon_name: str,
    horizon_start_days: int,
    horizon_end_days: int,
    base_df: pd.DataFrame,
    proteomics_df: pd.DataFrame,
    columns: ColumnNames,
    config: AnalysisConfig,
    global_train_indices: np.ndarray,
    global_test_indices: np.ndarray,
    klk3_values_all: np.ndarray,
    age_values_all: np.ndarray,
    klk3_median_train: float,
    age_feature_exclusions: set,
) -> Optional[Dict[str, Any]]:
    del horizon_start_days

    print("=" * 70)
    print(f"Horizon: {horizon_name} (0-{horizon_end_days} days)")
    print("=" * 70)

    followup_days = base_df[columns.followup_days].to_numpy(dtype=float)
    observed_event = base_df[columns.event_flag].to_numpy(dtype=int)

    label_positive = (observed_event == 1) & (followup_days <= horizon_end_days)
    label_negative = (observed_event == 0) & (followup_days > horizon_end_days)
    included_mask = label_positive | label_negative

    horizon_labels = label_positive.astype(int)
    included_indices = np.where(included_mask)[0]

    print(f"Included samples: {len(included_indices)}")
    print(f"Included events: {int(horizon_labels[included_indices].sum())}")

    if len(included_indices) < config.min_horizon_n:
        print("Skipping horizon due to low N.")
        return None

    train_indices = np.intersect1d(global_train_indices, included_indices)
    test_indices = np.intersect1d(global_test_indices, included_indices)

    y_train = horizon_labels[train_indices]
    y_test = horizon_labels[test_indices]

    print(f"Train/Test for horizon: {len(train_indices)} / {len(test_indices)}")

    if y_train.sum() < 2 or y_test.sum() < 2:
        print("Skipping horizon due to insufficient positive cases.")
        return None

    proteomics_feature_pool = [
        col
        for col in proteomics_df.columns
        if col in base_df.columns and col not in [columns.participant_id, columns.klk3_marker] and col not in age_feature_exclusions
    ]

    if len(proteomics_feature_pool) == 0:
        print("Skipping horizon: no proteomics features available after filtering.")
        return None

    x_train_feature_pool = base_df.loc[train_indices, proteomics_feature_pool].fillna(-1.0).to_numpy(dtype=np.float32)

    try:
        xgb_static = default_xgb_static_params(seed=config.random_seed, n_jobs=config.cv_jobs)
        preselect_model = XGBClassifier(**xgb_static)
        preselect_model.fit(x_train_feature_pool, y_train)

        shap_ranked_features = xgb_mean_abs_shap(preselect_model, x_train_feature_pool, proteomics_feature_pool)
        if len(shap_ranked_features) == 0:
            raise RuntimeError("SHAP preselection returned no features.")

        selected_proteomics_features = [
            feature
            for feature, _ in shap_ranked_features[: min(config.top_shap_features, len(shap_ranked_features))]
        ]
    except Exception as err:
        print(f"Skipping horizon due to SHAP preselection failure: {err}")
        return None

    if len(selected_proteomics_features) == 0:
        print("Skipping horizon: selected proteomics feature list is empty.")
        return None

    print(f"Selected proteomics features: {len(selected_proteomics_features)}")

    matrices = build_model_matrices(
        base_df=base_df,
        train_indices=train_indices,
        test_indices=test_indices,
        klk3_values_all=klk3_values_all,
        selected_proteomics_features=selected_proteomics_features,
    )

    model_scores, model_fit_ok, feature_importances_local, hyperparameters_local = fit_models_for_horizon(
        x_mats=matrices,
        y_train=y_train,
        y_test=y_test,
        proteomics_features=selected_proteomics_features,
        config=config,
    )

    if not all(model_fit_ok[m] for m in MODEL_ORDER):
        print("Skipping horizon: not all models were fit successfully.")
        return None

    typed_model_scores = {model_name: model_scores[model_name] for model_name in MODEL_ORDER}

    klk3_test = klk3_values_all[test_indices]
    age_test = age_values_all[test_indices]

    subgroup_masks = build_subgroup_masks(
        klk3_values=klk3_test,
        age_values=age_test,
        klk3_median=klk3_median_train,
        age_cutoff=config.age_cutoff_years,
    )

    auc_rows = collect_metric_rows(
        horizon_name=horizon_name,
        subgroup_masks=subgroup_masks,
        y_true=y_test,
        model_scores=typed_model_scores,
        metric_fn=roc_auc_score,
        metric_name="AUC_obs",
        n_bootstraps=config.bootstraps,
        rng_seed=config.random_seed,
        min_group_n=config.min_group_n,
    )

    auprc_rows = collect_metric_rows(
        horizon_name=horizon_name,
        subgroup_masks=subgroup_masks,
        y_true=y_test,
        model_scores=typed_model_scores,
        metric_fn=average_precision_score,
        metric_name="AUPRC_obs",
        n_bootstraps=config.bootstraps,
        rng_seed=config.random_seed,
        min_group_n=config.min_group_n,
    )

    biopsy_rows = run_biopsy_simulation(
        horizon_name=horizon_name,
        y_test=y_test,
        age_test=age_test,
        model_scores=typed_model_scores,
        age_cutoff=config.age_cutoff_years,
        min_group_n=config.min_group_n,
    )

    km_rows: List[Dict[str, Any]] = []
    test_followup_days = followup_days[test_indices]
    test_time_years = np.minimum(test_followup_days, horizon_end_days) / 365.25
    test_event = y_test.astype(int)

    for model_name in MODEL_ORDER:
        model_score = typed_model_scores[model_name]
        threshold = youden_best_threshold(test_event, model_score)
        if threshold is None:
            continue

        high_risk = model_score >= threshold
        low_risk = ~high_risk
        if high_risk.sum() == 0 or low_risk.sum() == 0:
            continue

        pvalue, _ = two_sample_logrank_pvalue(
            times_a=test_time_years[low_risk],
            events_a=test_event[low_risk],
            times_b=test_time_years[high_risk],
            events_b=test_event[high_risk],
        )

        km_rows.append(
            {
                "Horizon": horizon_name,
                "Model": model_name,
                "Threshold": float(threshold),
                "N_High": int(high_risk.sum()),
                "N_Low": int(low_risk.sum()),
                "Events_High": int(test_event[high_risk].sum()),
                "Events_Low": int(test_event[low_risk].sum()),
                "LogRank_p": pvalue,
            }
        )

    km_payload = {
        "time_years": test_time_years,
        "events": test_event,
        "age": age_test,
        "klk3": klk3_test,
        "xlim_start": 0.0,
        "xlim_end": horizon_end_days / 365.25,
        "scores": typed_model_scores,
    }

    return {
        "horizon": horizon_name,
        "auc_rows": auc_rows,
        "auprc_rows": auprc_rows,
        "biopsy_rows": biopsy_rows,
        "km_rows": km_rows,
        "km_payload": km_payload,
        "selected_features": selected_proteomics_features,
        "feature_importances": feature_importances_local,
        "hyperparameters": hyperparameters_local,
    }


def grouped_barplot_with_ci(
    metric_df: pd.DataFrame,
    horizon_name: str,
    subgroup_ids: List[str],
    subgroup_labels: List[str],
    metric_column: str,
    y_label: str,
    title: str,
    output_dir: str,
    filename_base: str,
    annotate_model: Optional[str] = "proteomics_only",
    y_max: Optional[float] = None,
) -> None:
    subset = metric_df[(metric_df["Horizon"] == horizon_name) & (metric_df["SubgroupId"].isin(subgroup_ids))].copy()
    if subset.empty:
        return

    pivot = subset.pivot(index="SubgroupId", columns="Model", values=metric_column).reindex(subgroup_ids)
    pivot_lo = subset.pivot(index="SubgroupId", columns="Model", values="CI_lo").reindex(subgroup_ids)
    pivot_hi = subset.pivot(index="SubgroupId", columns="Model", values="CI_hi").reindex(subgroup_ids)

    x = np.arange(len(subgroup_ids))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model_name in enumerate(MODEL_ORDER):
        if model_name not in pivot.columns:
            continue

        values = pivot[model_name].values.astype(float)
        lo_values = pivot_lo[model_name].values.astype(float)
        hi_values = pivot_hi[model_name].values.astype(float)

        lo_err = np.nan_to_num(np.maximum(0.0, values - lo_values), nan=0.0)
        hi_err = np.nan_to_num(np.maximum(0.0, hi_values - values), nan=0.0)

        bar_positions = x + i * width
        ax.bar(
            bar_positions,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            alpha=0.85,
            yerr=[lo_err, hi_err],
            capsize=5,
            error_kw={"elinewidth": 1.8},
        )

        if annotate_model == model_name:
            for j, (xj, value) in enumerate(zip(bar_positions, values)):
                subgroup_id = subgroup_ids[j]
                row = subset[(subset["SubgroupId"] == subgroup_id) & (subset["Model"] == model_name)]
                if row.empty:
                    continue
                n_val = int(row["N"].iloc[0])
                e_val = int(row["Events"].iloc[0])
                ax.text(
                    xj,
                    value - 0.05,
                    f"n={n_val}\nevents={e_val}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

    ax.set_xticks(x + (len(MODEL_ORDER) - 1) * width / 2)
    ax.set_xticklabels(subgroup_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if y_max is not None:
        ax.set_ylim(0.0, y_max)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, filename_base)


def plot_auc_and_auprc_summaries(
    auc_df: pd.DataFrame,
    auprc_df: pd.DataFrame,
    config: AnalysisConfig,
    klk3_median_train: float,
) -> None:
    if len(auc_df) > 0:
        for horizon_name, _, _ in config.horizons:
            grouped_barplot_with_ci(
                metric_df=auc_df,
                horizon_name=horizon_name,
                subgroup_ids=["klk3_le_median", "klk3_gt_median"],
                subgroup_labels=[f"KLK3 <= {klk3_median_train:.3f}", f"KLK3 > {klk3_median_train:.3f}"],
                metric_column="AUC_obs",
                y_label=f"AUC (test, B={config.bootstraps})",
                title=f"AUC by KLK3 median - {horizon_name}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUC_KLK3Groups_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )

            grouped_barplot_with_ci(
                metric_df=auc_df,
                horizon_name=horizon_name,
                subgroup_ids=["age_lt_cutoff_klk3_le_median", "age_lt_cutoff_klk3_gt_median"],
                subgroup_labels=[f"Age<{config.age_cutoff_years}: KLK3 <= median", f"Age<{config.age_cutoff_years}: KLK3 > median"],
                metric_column="AUC_obs",
                y_label=f"AUC (test, B={config.bootstraps})",
                title=f"AUC by KLK3 median - {horizon_name}, Age<{config.age_cutoff_years}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUC_AgeLT_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )

            grouped_barplot_with_ci(
                metric_df=auc_df,
                horizon_name=horizon_name,
                subgroup_ids=["age_ge_cutoff_klk3_le_median", "age_ge_cutoff_klk3_gt_median"],
                subgroup_labels=[f"Age>={config.age_cutoff_years}: KLK3 <= median", f"Age>={config.age_cutoff_years}: KLK3 > median"],
                metric_column="AUC_obs",
                y_label=f"AUC (test, B={config.bootstraps})",
                title=f"AUC by KLK3 median - {horizon_name}, Age>={config.age_cutoff_years}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUC_AgeGE_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )

    if len(auprc_df) > 0:
        for horizon_name, _, _ in config.horizons:
            grouped_barplot_with_ci(
                metric_df=auprc_df,
                horizon_name=horizon_name,
                subgroup_ids=["klk3_le_median", "klk3_gt_median"],
                subgroup_labels=[f"KLK3 <= {klk3_median_train:.3f}", f"KLK3 > {klk3_median_train:.3f}"],
                metric_column="AUPRC_obs",
                y_label=f"AUPRC (test, B={config.bootstraps})",
                title=f"AUPRC by KLK3 median - {horizon_name}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUPRC_KLK3Groups_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )

            grouped_barplot_with_ci(
                metric_df=auprc_df,
                horizon_name=horizon_name,
                subgroup_ids=["age_lt_cutoff_klk3_le_median", "age_lt_cutoff_klk3_gt_median"],
                subgroup_labels=[f"Age<{config.age_cutoff_years}: KLK3 <= median", f"Age<{config.age_cutoff_years}: KLK3 > median"],
                metric_column="AUPRC_obs",
                y_label=f"AUPRC (test, B={config.bootstraps})",
                title=f"AUPRC by KLK3 median - {horizon_name}, Age<{config.age_cutoff_years}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUPRC_AgeLT_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )

            grouped_barplot_with_ci(
                metric_df=auprc_df,
                horizon_name=horizon_name,
                subgroup_ids=["age_ge_cutoff_klk3_le_median", "age_ge_cutoff_klk3_gt_median"],
                subgroup_labels=[f"Age>={config.age_cutoff_years}: KLK3 <= median", f"Age>={config.age_cutoff_years}: KLK3 > median"],
                metric_column="AUPRC_obs",
                y_label=f"AUPRC (test, B={config.bootstraps})",
                title=f"AUPRC by KLK3 median - {horizon_name}, Age>={config.age_cutoff_years}",
                output_dir=config.output_dir,
                filename_base=f"Public_AUPRC_AgeGE_{horizon_name}",
                annotate_model="proteomics_only",
                y_max=1.0,
            )


def plot_biopsy_summaries(biopsy_df: pd.DataFrame, config: AnalysisConfig) -> None:
    if biopsy_df.empty:
        return

    overall_df = biopsy_df[biopsy_df["AgeGroup"].isna()].copy()
    if len(overall_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for metric_name, ax in zip(["TP", "FP", "FN"], axes):
            pivot = overall_df.pivot(index="Horizon", columns="Strategy", values=metric_name)
            pivot.plot(kind="bar", ax=ax, width=0.75, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Horizon")
            ax.set_title(f"{metric_name} by horizon (top 5% risk biopsy)")
            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        save_figure(fig, config.output_dir, "Public_Biopsy_TP_FP_FN")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for metric_name, ax in zip(["Sensitivity", "Specificity"], axes):
            pivot = overall_df.pivot(index="Horizon", columns="Strategy", values=metric_name)
            pivot.plot(kind="bar", ax=ax, width=0.75, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Horizon")
            ax.set_ylim([0.0, 1.05])
            ax.set_title(f"{metric_name} by horizon (top 5% risk biopsy)")
            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        save_figure(fig, config.output_dir, "Public_Biopsy_Sensitivity_Specificity")

        horizons = list(overall_df["Horizon"].unique())
        reduction_proteomics = []
        reduction_combined = []
        for horizon_name in horizons:
            sub = overall_df[overall_df["Horizon"] == horizon_name]
            fp_klk3 = sub[sub["Model"] == "klk3_only"]["FP"].values
            fp_proteomics = sub[sub["Model"] == "proteomics_only"]["FP"].values
            fp_combined = sub[sub["Model"] == "klk3_plus_proteomics"]["FP"].values
            if len(fp_klk3) == 0 or len(fp_proteomics) == 0 or len(fp_combined) == 0:
                continue

            klk3_fp_val = float(fp_klk3[0])
            proteomics_fp_val = float(fp_proteomics[0])
            combined_fp_val = float(fp_combined[0])

            if klk3_fp_val > 0:
                reduction_proteomics.append(((klk3_fp_val - proteomics_fp_val) / klk3_fp_val) * 100.0)
                reduction_combined.append(((klk3_fp_val - combined_fp_val) / klk3_fp_val) * 100.0)
            else:
                reduction_proteomics.append(0.0)
                reduction_combined.append(0.0)

        if len(reduction_proteomics) == len(horizons) and len(horizons) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(horizons))
            width = 0.35
            ax.bar(x - width / 2, reduction_proteomics, width=width, color="#ff7f0e", alpha=0.8, label="Proteomics vs KLK3")
            ax.bar(x + width / 2, reduction_combined, width=width, color="#2ca02c", alpha=0.8, label="KLK3+Proteomics vs KLK3")
            ax.axhline(y=0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(horizons)
            ax.set_ylabel("False positive reduction (%)")
            ax.set_xlabel("Horizon")
            ax.set_title("Reduction in unnecessary biopsies vs KLK3-only strategy")
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, config.output_dir, "Public_Biopsy_FP_Reduction")

    age_df = biopsy_df[biopsy_df["AgeGroup"].notna()].copy()
    if len(age_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, metric_name in enumerate(["TP", "FP", "Sensitivity", "Specificity"]):
            ax = axes[idx // 2, idx % 2]
            x_labels: List[str] = []
            x_pos_lookup: Dict[Tuple[str, str], int] = {}

            sorted_horizons = sorted(age_df["Horizon"].unique())
            sorted_age_groups = sorted(age_df["AgeGroup"].unique())

            counter = 0
            for horizon_name in sorted_horizons:
                for age_group_label in sorted_age_groups:
                    x_labels.append(f"{horizon_name}\n{age_group_label}")
                    x_pos_lookup[(horizon_name, age_group_label)] = counter
                    counter += 1

            for strategy_label in ["KLK3 only", "Proteomics", "KLK3+Proteomics"]:
                sub = age_df[age_df["Strategy"] == strategy_label]
                if sub.empty:
                    continue

                x_vals: List[int] = []
                y_vals: List[float] = []
                for horizon_name in sorted_horizons:
                    for age_group_label in sorted_age_groups:
                        row = sub[(sub["Horizon"] == horizon_name) & (sub["AgeGroup"] == age_group_label)]
                        if row.empty:
                            continue
                        x_vals.append(x_pos_lookup[(horizon_name, age_group_label)])
                        y_vals.append(float(row[metric_name].iloc[0]))

                if len(x_vals) > 0:
                    ax.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=6, label=strategy_label)

            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Horizon and age group")
            ax.set_title(f"{metric_name} by horizon and age group")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        save_figure(fig, config.output_dir, "Public_Biopsy_AgeStratified")


def plot_klk3_age_exploration(
    klk3_values_test_global: np.ndarray,
    age_values_test_global: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    age_bins = [0, 50, 60, 70, 150]
    age_labels = ["<50", "50-60", "60-70", "70+"]
    age_groups = pd.cut(age_values_test_global, bins=age_bins, labels=age_labels, include_lowest=True)

    explore_df = pd.DataFrame({"KLK3": klk3_values_test_global, "Age": age_values_test_global, "AgeGroup": age_groups})
    explore_df = explore_df[explore_df["KLK3"] > 0].copy()

    if len(explore_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        explore_df.boxplot(column="KLK3", by="AgeGroup", ax=axes[0])
        axes[0].set_xlabel("Age group")
        axes[0].set_ylabel("KLK3")
        axes[0].set_title("KLK3 distribution by age group")
        plt.sca(axes[0])
        plt.xticks(rotation=0)

        sns.violinplot(data=explore_df, x="AgeGroup", y="KLK3", ax=axes[1])
        axes[1].set_xlabel("Age group")
        axes[1].set_ylabel("KLK3")
        axes[1].set_title("KLK3 distribution by age group (violin)")

        plt.tight_layout()
        save_figure(fig, output_dir, "Public_Exploratory_KLK3_AgeGroup")

    print("\nKLK3 summary by age group (global test split):")
    for label in age_labels:
        subset = explore_df.loc[explore_df["AgeGroup"] == label, "KLK3"]
        if len(subset) == 0:
            continue
        print(
            f"  {label:8s}: N={len(subset):4d}, mean={subset.mean():.3f}, "
            f"median={subset.median():.3f}, std={subset.std():.3f}"
        )

    return explore_df


def assign_quintile_labels(scores: np.ndarray, q_edges: np.ndarray) -> np.ndarray:
    labels = np.full(len(scores), 5, dtype=int)
    labels[scores <= q_edges[0]] = 1
    labels[(scores > q_edges[0]) & (scores <= q_edges[1])] = 2
    labels[(scores > q_edges[1]) & (scores <= q_edges[2])] = 3
    labels[(scores > q_edges[2]) & (scores <= q_edges[3])] = 4
    return labels


def plot_km_quintiles(
    times: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    quintile_edges: np.ndarray,
    title: str,
    subtitle: str,
    output_dir: str,
    filename_base: str,
    min_group_n: int,
    xlim_start: float,
    xlim_end: float,
) -> Optional[Dict[str, Any]]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Diagnosis-free survival probability")
    ax.set_ylim(0.8, 1.02)
    ax.set_xlim(xlim_start, xlim_end)

    if len(times) < min_group_n:
        ax.text(0.5, 0.60, f"Insufficient N in subgroup (N={len(times)})", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.52, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, filename_base)
        return None

    if len(quintile_edges) != 4:
        ax.text(0.5, 0.60, "Quintile thresholds unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.52, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, filename_base)
        return None

    masks = [
        scores <= quintile_edges[0],
        (scores > quintile_edges[0]) & (scores <= quintile_edges[1]),
        (scores > quintile_edges[1]) & (scores <= quintile_edges[2]),
        (scores > quintile_edges[2]) & (scores <= quintile_edges[3]),
        scores > quintile_edges[3],
    ]

    palette = sns.color_palette("viridis", 5)

    grouped_times: List[np.ndarray] = []
    grouped_events: List[np.ndarray] = []
    quintile_counts: List[Tuple[int, int]] = []
    plotted_any = False

    for idx, mask in enumerate(masks):
        if mask.sum() == 0:
            quintile_counts.append((0, 0))
            continue

        t_group = times[mask]
        e_group = events[mask]
        x_vals, y_vals = km_step(t_group, e_group)

        ax.step(
            x_vals,
            y_vals,
            where="post",
            linewidth=2,
            color=palette[idx],
            label=f"Q{idx + 1} (N={len(t_group)}, E={int(e_group.sum())})",
        )
        plotted_any = True

        quintile_counts.append((len(t_group), int(e_group.sum())))
        if len(t_group) >= 2:
            grouped_times.append(t_group)
            grouped_events.append(e_group)

    if not plotted_any:
        ax.text(0.5, 0.60, "No data available in quintile groups", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.52, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, filename_base)
        return None

    pvalue = np.nan
    k_used = 0
    if len(grouped_times) < 2:
        logrank_note = "Log-rank p=NA (insufficient groups)"
    else:
        pvalue, _, k_used = k_sample_logrank_pvalue(grouped_times, grouped_events)
        if np.isnan(pvalue):
            logrank_note = "Log-rank p=NA (computation failed)"
        else:
            logrank_note = f"Log-rank p={pvalue:.3g} (k={k_used})"

    ax.text(0.02, 0.02, logrank_note, transform=ax.transAxes, fontsize=9, va="bottom")
    ax.text(0.02, 0.10, subtitle, transform=ax.transAxes, fontsize=9, va="bottom")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, filename_base)

    return {
        "pvalue": pvalue,
        "k_used": k_used,
        "quintile_counts": quintile_counts,
        "quintile_edges": quintile_edges.tolist(),
    }


def generate_km_quintile_outputs(
    km_plot_data: Dict[str, Dict[str, Any]],
    config: AnalysisConfig,
    klk3_median_train: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    quintile_summary_rows: List[Dict[str, Any]] = []
    quintile_label_rows: List[Dict[str, Any]] = []

    for horizon_name, payload in km_plot_data.items():
        time_years = np.asarray(payload["time_years"], dtype=float)
        event_labels = np.asarray(payload["events"], dtype=int)
        age_values = np.asarray(payload["age"], dtype=float)
        klk3_values = np.asarray(payload["klk3"], dtype=float)
        xlim_start = float(payload["xlim_start"])
        xlim_end = float(payload["xlim_end"])

        if len(time_years) == 0:
            continue

        population_masks = [
            ("AllPatients", np.ones(len(time_years), dtype=bool), "All patients"),
            ("AgeLT", age_values < config.age_cutoff_years, f"Age<{config.age_cutoff_years}"),
            ("AgeGE", age_values >= config.age_cutoff_years, f"Age>={config.age_cutoff_years}"),
        ]
        klk3_masks = [
            ("KLK3le", klk3_values <= klk3_median_train, f"KLK3 <= {klk3_median_train:.3f}"),
            ("KLK3gt", klk3_values > klk3_median_train, f"KLK3 > {klk3_median_train:.3f}"),
        ]

        for model_name, full_scores in payload["scores"].items():
            scores_all = np.asarray(full_scores, dtype=float)
            q_edges = np.percentile(scores_all, [20, 40, 60, 80])
            labels_all = assign_quintile_labels(scores_all, q_edges)

            for idx, q_label in enumerate(labels_all.tolist()):
                quintile_label_rows.append(
                    {
                        "Horizon": horizon_name,
                        "Model": model_name,
                        "idx": idx,
                        "Quintile": int(q_label),
                    }
                )

            for pop_id, pop_mask, pop_title in population_masks:
                for klk3_id, klk3_mask, klk3_title in klk3_masks:
                    subgroup_mask = pop_mask & klk3_mask
                    if subgroup_mask.sum() < config.min_group_n:
                        continue

                    t_sub = time_years[subgroup_mask]
                    e_sub = event_labels[subgroup_mask]
                    s_sub = scores_all[subgroup_mask]

                    title = (
                        f"Kaplan-Meier: {MODEL_LABELS.get(model_name, model_name)} - "
                        f"{horizon_name}, {pop_title}, {klk3_title}"
                    )
                    subtitle = "Quintile risk groups based on full test split"
                    filename_base = f"Public_KM_{model_name}_{pop_id}_{klk3_id}_{horizon_name}"

                    result = plot_km_quintiles(
                        times=t_sub,
                        events=e_sub,
                        scores=s_sub,
                        quintile_edges=q_edges,
                        title=title,
                        subtitle=subtitle,
                        output_dir=config.output_dir,
                        filename_base=filename_base,
                        min_group_n=config.min_group_n,
                        xlim_start=xlim_start,
                        xlim_end=xlim_end,
                    )

                    if result is None:
                        continue

                    q_counts = result["quintile_counts"]
                    quintile_summary_rows.append(
                        {
                            "Horizon": horizon_name,
                            "Model": model_name,
                            "Population": pop_id,
                            "KLK3_Group": klk3_id,
                            "XlimStart": xlim_start,
                            "XlimEnd": xlim_end,
                            "Q20": result["quintile_edges"][0],
                            "Q40": result["quintile_edges"][1],
                            "Q60": result["quintile_edges"][2],
                            "Q80": result["quintile_edges"][3],
                            "LogRank_p": result["pvalue"],
                            "k_used": result["k_used"],
                            "Q1_N": q_counts[0][0],
                            "Q1_E": q_counts[0][1],
                            "Q2_N": q_counts[1][0],
                            "Q2_E": q_counts[1][1],
                            "Q3_N": q_counts[2][0],
                            "Q3_E": q_counts[2][1],
                            "Q4_N": q_counts[3][0],
                            "Q4_E": q_counts[3][1],
                            "Q5_N": q_counts[4][0],
                            "Q5_E": q_counts[4][1],
                        }
                    )

    return quintile_summary_rows, quintile_label_rows


def plot_feature_importance_panels(
    horizon_feature_importance: Dict[Tuple[str, str], List[Tuple[str, float]]],
    config: AnalysisConfig,
) -> None:
    for horizon_name, _, _ in config.horizons:
        for model_name in ["proteomics_only", "klk3_plus_proteomics"]:
            key = (horizon_name, model_name)
            if key not in horizon_feature_importance:
                continue

            ranked = horizon_feature_importance[key]
            if ranked is None or len(ranked) == 0:
                continue

            df_ranked = pd.DataFrame(ranked, columns=["Feature", "MeanAbsSHAP"]).copy()
            df_ranked["Feature"] = df_ranked["Feature"].str.replace("PSA", "KLK3", regex=False)
            df_ranked["Feature"] = df_ranked["Feature"].str.replace("TEU_BaC_AgeAtRec", "Age", regex=False)
            df_ranked = df_ranked.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

            top_n = min(20, len(df_ranked))
            fig = plt.figure(figsize=(6, max(3, top_n * 0.25)))
            sns.barplot(x="MeanAbsSHAP", y="Feature", data=df_ranked.head(top_n), orient="h")
            plt.title(f"{horizon_name} - {MODEL_LABELS[model_name]} SHAP importance (top {top_n})")
            filename = f"Public_{horizon_name}_{model_name}_feature_importance_top{top_n}"
            save_figure(fig, config.output_dir, filename)


def flatten_feature_importance_rows(
    horizon_feature_importance: Dict[Tuple[str, str], List[Tuple[str, float]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (horizon_name, model_name), feature_list in horizon_feature_importance.items():
        for feature_name, mean_abs_shap in feature_list:
            rows.append(
                {
                    "Horizon": horizon_name,
                    "Model": model_name,
                    "Feature": feature_name,
                    "MeanAbsSHAP": mean_abs_shap,
                }
            )
    return rows


def save_km_reproducibility_tables(
    km_plot_data: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> None:
    base_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []

    for horizon_name, payload in km_plot_data.items():
        time_years = np.asarray(payload.get("time_years", []), dtype=float)
        events = np.asarray(payload.get("events", []), dtype=int)
        age_values = np.asarray(payload.get("age", []), dtype=float)
        klk3_values = np.asarray(payload.get("klk3", []), dtype=float)

        n_samples = len(time_years)
        if n_samples == 0:
            continue

        for i in range(n_samples):
            base_rows.append(
                {
                    "Horizon": horizon_name,
                    "idx": i,
                    "time_years": time_years[i],
                    "event": int(events[i]),
                    "age": age_values[i],
                    "klk3": klk3_values[i],
                }
            )

        for model_name, scores in payload.get("scores", {}).items():
            scores = np.asarray(scores, dtype=float)
            for i in range(n_samples):
                score_rows.append(
                    {
                        "Horizon": horizon_name,
                        "idx": i,
                        "model": model_name,
                        "score": scores[i],
                    }
                )

    if len(base_rows) > 0:
        save_dataframe(pd.DataFrame(base_rows), output_dir, "km_plot_base.csv")
    if len(score_rows) > 0:
        save_dataframe(pd.DataFrame(score_rows), output_dir, "km_plot_scores.csv")


def main() -> None:
    start_time = time.time()
    args = parse_args()

    config = build_config_from_args(args)
    columns = ColumnNames()

    validate_and_prepare_paths(config)

    sns.set(style="whitegrid")

    proteomics_df, demographics_df, outcomes_df = load_input_tables(config)
    merged_cohort_df = merge_and_clean_tables(
        proteomics_df=proteomics_df,
        demographics_df=demographics_df,
        outcomes_df=outcomes_df,
        columns=columns,
    )

    base_df = prepare_base_population(merged_cohort_df, columns)
    print(f"Base population N: {len(base_df)}, events: {int(base_df[columns.event_flag].sum())}")

    age_candidates = find_age_columns(base_df, columns)
    age_column = choose_age_column(age_candidates, columns)
    print(f"Using age column: {age_column}")

    y_base = base_df[columns.event_flag].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
    global_train_indices, global_test_indices = next(splitter.split(np.zeros(len(y_base)), y_base))
    print(f"Global train/test split: {len(global_train_indices)} / {len(global_test_indices)}")

    klk3_values_all = base_df[columns.klk3_marker].fillna(-1.0).to_numpy(dtype=float)
    age_values_all = base_df[age_column].fillna(-1.0).to_numpy(dtype=float)

    klk3_train = klk3_values_all[global_train_indices]
    klk3_train_nonmissing = klk3_train[(~np.isnan(klk3_train)) & (klk3_train != -1.0)]
    if len(klk3_train_nonmissing) == 0:
        raise ValueError("No valid KLK3 values found in training split.")
    klk3_median_train = float(np.median(klk3_train_nonmissing))
    print(f"Training KLK3 median: {klk3_median_train:.6f}")

    age_feature_exclusions = set(age_candidates)
    age_feature_exclusions.update(["BaC_Age_x", "TEU_BaC_AgeAtRec", "Age"])

    auc_rows_all: List[Dict[str, Any]] = []
    auprc_rows_all: List[Dict[str, Any]] = []
    biopsy_rows_all: List[Dict[str, Any]] = []
    km_rows_all: List[Dict[str, Any]] = []
    hyperparameter_rows_all: List[Dict[str, Any]] = []

    horizon_feature_importance: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    km_plot_data: Dict[str, Dict[str, Any]] = {}

    for horizon_name, horizon_start_days, horizon_end_days in config.horizons:
        result = run_horizon_analysis(
            horizon_name=horizon_name,
            horizon_start_days=horizon_start_days,
            horizon_end_days=horizon_end_days,
            base_df=base_df,
            proteomics_df=proteomics_df,
            columns=columns,
            config=config,
            global_train_indices=global_train_indices,
            global_test_indices=global_test_indices,
            klk3_values_all=klk3_values_all,
            age_values_all=age_values_all,
            klk3_median_train=klk3_median_train,
            age_feature_exclusions=age_feature_exclusions,
        )

        if result is None:
            continue

        auc_rows_all.extend(result["auc_rows"])
        auprc_rows_all.extend(result["auprc_rows"])
        biopsy_rows_all.extend(result["biopsy_rows"])
        km_rows_all.extend(result["km_rows"])

        for local_row in result["hyperparameters"]:
            row = {"Horizon": horizon_name}
            row.update(local_row)
            hyperparameter_rows_all.append(row)

        km_plot_data[horizon_name] = result["km_payload"]

        local_feature_importance: Dict[Tuple[str, str], List[Tuple[str, float]]] = result["feature_importances"]
        for (model_name, _subset), ranked_features in local_feature_importance.items():
            horizon_feature_importance[(horizon_name, model_name)] = ranked_features

    auc_df = pd.DataFrame(auc_rows_all)
    auprc_df = pd.DataFrame(auprc_rows_all)
    biopsy_df = pd.DataFrame(biopsy_rows_all)
    km_df = pd.DataFrame(km_rows_all)
    hyperparameter_df = pd.DataFrame(hyperparameter_rows_all)

    if len(auc_df) > 0:
        save_dataframe(auc_df, config.output_dir, "results_AUC.csv")
    if len(auprc_df) > 0:
        save_dataframe(auprc_df, config.output_dir, "results_AUPRC.csv")
    if len(biopsy_df) > 0:
        save_dataframe(biopsy_df, config.output_dir, "biopsy_results.csv")
    if len(km_df) > 0:
        save_dataframe(km_df, config.output_dir, "km_results.csv")

    medians_df = pd.DataFrame(
        [
            {
                "klk3_median_train": klk3_median_train,
                "age_column_used": age_column,
                "age_cutoff_years": config.age_cutoff_years,
            }
        ]
    )
    save_dataframe(medians_df, config.output_dir, "klk3_medians.csv")

    if len(hyperparameter_df) > 0:
        save_dataframe(hyperparameter_df, config.output_dir, "best_hyperparameters.csv")
        print("\nSelected hyperparameters:")
        print(hyperparameter_df.to_string(index=False))

    feature_importance_rows = flatten_feature_importance_rows(horizon_feature_importance)
    if len(feature_importance_rows) > 0:
        save_dataframe(pd.DataFrame(feature_importance_rows), config.output_dir, "feature_importances.csv")

    save_km_reproducibility_tables(km_plot_data=km_plot_data, output_dir=config.output_dir)

    print("\nGenerating plots...")
    plot_auc_and_auprc_summaries(auc_df=auc_df, auprc_df=auprc_df, config=config, klk3_median_train=klk3_median_train)
    plot_biopsy_summaries(biopsy_df=biopsy_df, config=config)

    global_klk3_test = klk3_values_all[global_test_indices]
    global_age_test = age_values_all[global_test_indices]
    explore_df = plot_klk3_age_exploration(global_klk3_test, global_age_test, config.output_dir)
    if len(explore_df) > 0:
        save_dataframe(explore_df[["KLK3", "Age"]].copy(), config.output_dir, "klk3_age_testset.csv")

    quintile_summary_rows, quintile_label_rows = generate_km_quintile_outputs(
        km_plot_data=km_plot_data,
        config=config,
        klk3_median_train=klk3_median_train,
    )

    if len(quintile_summary_rows) > 0:
        save_dataframe(pd.DataFrame(quintile_summary_rows), config.output_dir, "km_quintile_summary.csv")
    if len(quintile_label_rows) > 0:
        save_dataframe(pd.DataFrame(quintile_label_rows), config.output_dir, "km_quintile_labels.csv")

    plot_feature_importance_panels(horizon_feature_importance, config)

    elapsed_sec = time.time() - start_time
    print("=" * 70)
    print(f"All results saved to: {config.output_dir}")
    print(f"Completed in {elapsed_sec:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
