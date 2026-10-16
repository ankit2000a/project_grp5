"""
Component C (Group 5) - Statistical Layer: Week 2 Implementation
CSIT332 Principles of Machine Learning Semester Project

This module implements the Week 2 statistical evaluation layer:
1. Winner-correct rate (Version A: Per-Sample Truth vs Version B: Asymptotic Truth)
2. Mean Kendall's tau (Version A vs Version B)
3. Mean optimism (Version A vs Version B)
4. Power & Sign-error rate via paired t-tests on chosen classifier pairs
   (Close gap < 0.01 vs Wide gap > 0.05 across all sample sizes)

==============================================================================
THE WEEK 2 PROBLEM AND CORRECTION
==============================================================================
In Week 1, optimism at small sample sizes (n=50) was computed against the
abundant-data (asymptotic) benchmark, producing a negative optimism value (-0.035).
While mathematically possible as a simple difference, a negative value cannot occur
if the metric is measuring pure selection optimism. This occurred because two distinct
effects were conflated:
1. Selection bias (optimism): The declared winner's observed score is inflated
   because it was selected as the maximum of several noisy estimators (positive effect).
2. Finite-sample training deficit: A model trained on a small sample (e.g. 40 rows in 5-fold CV)
   has lower true skill than the same model trained on abundant data (12,600 rows).

In Week 2, ground truth is measured at EACH sample size n_sub (50, 100, 200, 500)
by fitting models on the exact CV training folds and testing on a frozen held-out test
set (6,000 rows). Truth is also measured with abundant data ('asymptotic').

- Version A (Truth at sample size n): Measures the uncertainty a researcher actually faces.
  Optimism is positive and shrinks as sample size grows; winner-correct rate rises.
- Version B (Asymptotic truth): Conflates selection bias with sample-size deficit.
  Optimism is negative; winner-correct rate is penalized because different models
  are optimal at small vs large data regimes (e.g. logreg at n=50 vs gb at asymptotic).
==============================================================================
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def _resolve_filepath(path: str) -> str:
    """Resolve file path relative to current working directory or script directory."""
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_datasets(
    truth_source: Union[str, pd.DataFrame] = "reference_truth.csv",
    results_source: Union[str, pd.DataFrame] = "reference_results.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ground truth and results CSV files or dataframes.
    Normalizes whitespace and coerces numeric and categorical columns.
    Handles the 4-column truth format: [dataset_id, classifier_id, n_sub, true_auc].
    """
    if isinstance(truth_source, str):
        truth_source = _resolve_filepath(truth_source)
    if isinstance(results_source, str):
        results_source = _resolve_filepath(results_source)

    truth_df = (
        pd.read_csv(truth_source)
        if isinstance(truth_source, str)
        else truth_source.copy()
    )
    results_df = (
        pd.read_csv(results_source)
        if isinstance(results_source, str)
        else results_source.copy()
    )

    # Clean string identifiers
    truth_df["dataset_id"] = truth_df["dataset_id"].astype(str).str.strip()
    truth_df["classifier_id"] = truth_df["classifier_id"].astype(str).str.strip()
    if "n_sub" in truth_df.columns:
        truth_df["n_sub"] = truth_df["n_sub"].astype(str).str.strip()

    results_df["dataset_id"] = results_df["dataset_id"].astype(str).str.strip()
    results_df["classifier_id"] = results_df["classifier_id"].astype(str).str.strip()
    if "protocol_id" in results_df.columns:
        results_df["protocol_id"] = results_df["protocol_id"].astype(str).str.strip()

    # Coerce numeric fields
    truth_df["true_auc"] = pd.to_numeric(truth_df["true_auc"], errors="coerce")
    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    results_df["fold_id"] = pd.to_numeric(results_df["fold_id"], errors="coerce")
    if "n_sub" in results_df.columns:
        results_df["n_sub"] = pd.to_numeric(results_df["n_sub"], errors="coerce")
    if "replicate_id" in results_df.columns:
        results_df["replicate_id"] = pd.to_numeric(results_df["replicate_id"], errors="coerce")

    return truth_df, results_df


def get_truth_subset(
    truth_df: pd.DataFrame,
    dataset_id: str,
    n_sub: Optional[Union[int, str]] = None,
    mode: str = "per_sample",
) -> pd.DataFrame:
    """
    Extract relevant slice of ground truth for a given dataset and sample size.
    
    Parameters:
        truth_df: The full ground truth DataFrame.
        dataset_id: Target dataset ID.
        n_sub: Sample size (e.g. 50, 100, 200, 500).
        mode: 'per_sample' (Version A: truth at sample size n) or
              'asymptotic' (Version B: truth at abundant-data limit).
    """
    sub = truth_df[truth_df["dataset_id"] == dataset_id]
    if "n_sub" in sub.columns:
        if mode == "asymptotic":
            sub = sub[sub["n_sub"].str.lower() == "asymptotic"]
        else:
            if n_sub is not None:
                sub = sub[sub["n_sub"] == str(n_sub)]
    return sub


def validate_replicate(
    replicate_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    n_sub: Optional[int] = None,
    expected_folds: int = 5,
    required_classifiers: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate whether a replicate satisfies all integrity and edge-case rules:
    - Rule 4: Matches on dataset_id and n_sub. Every classifier appearing in results
      must have a valid, non-NaN true_auc in truth.csv.
    - Rule 3: Every required classifier must have exactly expected_folds rows with
      integer fold IDs matching {1, ..., expected_folds}.
    - Rules 2 & 5: Non-computable / NaN fold AUC or fold ID invalidates the replicate.
    """
    if replicate_df.empty:
        return False, "Replicate dataframe is empty."

    dataset_ids = replicate_df["dataset_id"].dropna().unique()
    if len(dataset_ids) != 1:
        return False, f"Multiple or missing dataset_ids in replicate: {dataset_ids}"
    dataset_id = str(dataset_ids[0])

    if n_sub is None and "n_sub" in replicate_df.columns:
        n_vals = replicate_df["n_sub"].dropna().unique()
        if len(n_vals) == 1:
            n_sub = int(n_vals[0])

    truth_subset = get_truth_subset(truth_df, dataset_id, n_sub=n_sub, mode="per_sample")
    if truth_subset.empty:
        # Fallback to dataset_id filter if n_sub column not in truth
        truth_subset = truth_df[truth_df["dataset_id"] == dataset_id]
        if truth_subset.empty:
            return False, f"Rule 4: No ground truth entries found for dataset '{dataset_id}'."

    replicate_clfs = set(replicate_df["classifier_id"].dropna().unique())
    for clf in replicate_clfs:
        clf_truth = truth_subset[truth_subset["classifier_id"] == clf]
        if clf_truth.empty:
            return False, f"Rule 4: Classifier '{clf}' has no matching entry in truth.csv."
        if pd.isna(clf_truth["true_auc"].iloc[0]):
            return False, f"Rule 4: Classifier '{clf}' in truth.csv has missing/NaN true_auc."

    if required_classifiers is None:
        required_classifiers = sorted(truth_subset["classifier_id"].dropna().unique().tolist())

    missing_required = set(required_classifiers) - replicate_clfs
    if missing_required:
        return False, f"Rule 3: Missing required classifiers: {sorted(list(missing_required))}."

    expected_fold_ids = set(range(1, expected_folds + 1))
    for clf in required_classifiers:
        clf_rows = replicate_df[replicate_df["classifier_id"] == clf]
        if len(clf_rows) != expected_folds:
            return False, f"Rule 3: Classifier '{clf}' has {len(clf_rows)} rows, expected {expected_folds}."
        if clf_rows["auc"].isna().any():
            return False, f"Rules 2 & 5: Classifier '{clf}' contains NaN/non-computable fold AUC."
        if clf_rows["fold_id"].isna().any():
            return False, f"Rules 2 & 5: Classifier '{clf}' contains NaN fold_id."

        fold_id_vals = clf_rows["fold_id"].to_numpy(dtype=float)
        if not np.all(np.equal(np.mod(fold_id_vals, 1), 0)):
            return False, f"Rule 3: Classifier '{clf}' has non-integer fold IDs."

        actual_fold_ids = set(int(x) for x in fold_id_vals)
        if actual_fold_ids != expected_fold_ids:
            return False, f"Rule 3: Classifier '{clf}' has invalid fold IDs {sorted(list(actual_fold_ids))}."

    return True, "Valid replicate."


def calculate_replicate_means(replicate_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each classifier's replicate-level observed CV AUC (mean across folds)."""
    summary = (
        replicate_df.groupby(["dataset_id", "classifier_id"])
        .agg(
            mean_auc=("auc", "mean"),
            fold_count=("auc", "count"),
            fold_aucs=("auc", list),
            fold_ids=("fold_id", list),
        )
        .reset_index()
    )
    return summary


def get_declared_winner(replicate_summary: pd.DataFrame) -> Tuple[str, float]:
    """
    Determine the declared winner based on highest observed mean AUC.
    Rule 1 (Exact tie in empirical mean AUC): Deterministically broken by selecting
    the lexicographically smallest classifier_id (ascending order).
    """
    sorted_df = replicate_summary.sort_values(
        by=["mean_auc", "classifier_id"], ascending=[False, True]
    )
    winner_row = sorted_df.iloc[0]
    return str(winner_row["classifier_id"]), float(winner_row["mean_auc"])


def get_true_best_classifiers(truth_subset: pd.DataFrame) -> Tuple[List[str], float, bool]:
    """
    Determine genuinely best classifier(s) from a truth subset.
    Truth-Tie Rule: Returns all classifiers sharing the maximum true AUC.
    """
    if truth_subset.empty:
        raise ValueError("Ground truth subset is empty.")
    max_true_auc = float(truth_subset["true_auc"].max())
    best_clfs = sorted(
        truth_subset[np.isclose(truth_subset["true_auc"], max_true_auc, atol=1e-9)][
            "classifier_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    is_truth_tie = len(best_clfs) > 1
    return best_clfs, max_true_auc, is_truth_tie


def compute_winner_correct(
    replicate_summary: pd.DataFrame,
    truth_subset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Winner-Correct indicator:
    - Declared winner is compared against the classifier(s) with highest true AUC in truth_subset.
    - Truth-Tie Rule: If declared winner is any of the tied true best classifiers, correct = 1.
    """
    declared_winner, declared_mean_auc = get_declared_winner(replicate_summary)
    true_bests, true_best_auc, is_truth_tie = get_true_best_classifiers(truth_subset)
    is_correct = 1 if declared_winner in true_bests else 0
    return {
        "is_correct": is_correct,
        "declared_winner": declared_winner,
        "declared_mean_auc": declared_mean_auc,
        "true_best": ", ".join(true_bests),
        "true_best_classifiers": true_bests,
        "true_best_auc": true_best_auc,
        "is_truth_tie": is_truth_tie,
    }


def compute_kendall_tau(
    replicate_summary: pd.DataFrame,
    truth_subset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Kendall's tau rank correlation between observed CV mean AUC and true AUC.
    Uses Kendall's tau-b (adjusts for ties in true or observed scores).
    """
    merged = pd.merge(
        replicate_summary, truth_subset, on="classifier_id"
    )
    merged = merged.sort_values(by="classifier_id").reset_index(drop=True)
    tau_res = stats.kendalltau(merged["true_auc"], merged["mean_auc"])
    tau_val = float(tau_res.statistic) if not np.isnan(tau_res.statistic) else 0.0
    return {
        "tau": tau_val,
        "p_value": float(tau_res.pvalue) if not np.isnan(tau_res.pvalue) else 1.0,
    }


def compute_optimism(
    replicate_summary: pd.DataFrame,
    truth_subset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Optimism:
    optimism = observed CV mean AUC of declared winner - true AUC of THAT SAME winner.
    """
    declared_winner, declared_mean_auc = get_declared_winner(replicate_summary)
    winner_row = truth_subset[truth_subset["classifier_id"] == declared_winner]
    if winner_row.empty:
        raise ValueError(f"Declared winner '{declared_winner}' not found in ground truth subset.")
    winner_true_auc = float(winner_row["true_auc"].iloc[0])
    optimism = declared_mean_auc - winner_true_auc
    return {
        "declared_winner": declared_winner,
        "declared_mean_auc": declared_mean_auc,
        "winner_true_auc": winner_true_auc,
        "optimism": float(optimism),
    }


def compute_paired_ttest(
    replicate_df: pd.DataFrame,
    truth_subset: pd.DataFrame,
    c1_id: str,
    c2_id: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform paired t-test for pair (c1 vs c2) across matched 5 CV folds.
    
    Evaluates detection outcome against ground truth:
    - If true_auc(c1) > true_auc(c2): c1 is genuinely superior.
      Significant rejection favoring c1 (mean_diff > 0, p < alpha) is CORRECT DETECTION (Power=1).
      Significant rejection favoring c2 (mean_diff < 0, p < alpha) is SIGN ERROR (SignError=1).
    - If true_auc(c2) > true_auc(c1): c2 is genuinely superior.
      Significant rejection favoring c2 (mean_diff < 0, p < alpha) is CORRECT DETECTION (Power=1).
      Significant rejection favoring c1 (mean_diff > 0, p < alpha) is SIGN ERROR (SignError=1).
    - If true_auc(c1) == true_auc(c2): Truth-Tie (H0 is true in reality).
      Any rejection (p < alpha) is a TYPE I ERROR / False Positive (Type1Error=1, Power=0).
    - Rules 6, 7, 8: Degenerate t-test, zero variance, p >= alpha, or mean_diff == 0 -> NO DETECTION.
    """
    c1_folds = replicate_df[replicate_df["classifier_id"] == c1_id].sort_values(by="fold_id")
    c2_folds = replicate_df[replicate_df["classifier_id"] == c2_id].sort_values(by="fold_id")

    c1_truth = truth_subset[truth_subset["classifier_id"] == c1_id]
    c2_truth = truth_subset[truth_subset["classifier_id"] == c2_id]
    c1_true_auc = float(c1_truth["true_auc"].iloc[0]) if not c1_truth.empty else np.nan
    c2_true_auc = float(c2_truth["true_auc"].iloc[0]) if not c2_truth.empty else np.nan

    is_truth_tie = (
        not np.isnan(c1_true_auc)
        and not np.isnan(c2_true_auc)
        and np.isclose(c1_true_auc, c2_true_auc, atol=1e-9)
    )

    if (
        c1_folds.empty
        or c2_folds.empty
        or len(c1_folds) < 2
        or len(c1_folds) != len(c2_folds)
        or list(c1_folds["fold_id"]) != list(c2_folds["fold_id"])
    ):
        return {
            "power": 0, "sign_error": 0, "type1_error": 0,
            "mean_difference": 0.0, "p_value": 1.0, "t_statistic": np.nan,
            "outcome": "NO DETECTION (Rule 6: Insufficient/unmatched folds)"
        }

    c1_vals = c1_folds["auc"].to_numpy(dtype=float)
    c2_vals = c2_folds["auc"].to_numpy(dtype=float)
    diffs = c1_vals - c2_vals
    mean_diff = float(np.mean(diffs))

    if np.isclose(mean_diff, 0.0, atol=1e-12):
        return {
            "power": 0, "sign_error": 0, "type1_error": 0,
            "mean_difference": 0.0, "p_value": 1.0, "t_statistic": 0.0,
            "outcome": "NO DETECTION (Rule 8: Mean difference = 0.0)"
        }

    diff_std = float(np.std(diffs, ddof=1))
    if np.isclose(diff_std, 0.0, atol=1e-12):
        return {
            "power": 0, "sign_error": 0, "type1_error": 0,
            "mean_difference": mean_diff, "p_value": np.nan, "t_statistic": np.nan,
            "outcome": "NO DETECTION (Rule 6: Zero variance in differences)"
        }

    try:
        ttest_res = stats.ttest_rel(c1_vals, c2_vals)
        t_stat = float(ttest_res.statistic)
        p_val = float(ttest_res.pvalue)
    except Exception:
        t_stat, p_val = np.nan, 1.0

    if np.isnan(p_val) or np.isnan(t_stat) or np.isinf(t_stat):
        return {
            "power": 0, "sign_error": 0, "type1_error": 0,
            "mean_difference": mean_diff, "p_value": np.nan, "t_statistic": np.nan,
            "outcome": "NO DETECTION (Rule 6: Degenerate t-test)"
        }

    power = 0
    sign_error = 0
    type1_error = 0

    if p_val < alpha:
        if is_truth_tie:
            type1_error = 1
            outcome = f"TYPE I ERROR (Truth Tie: true_auc({c1_id}) == true_auc({c2_id}), p < {alpha})"
        elif np.isnan(c1_true_auc) or np.isnan(c2_true_auc) or c1_true_auc > c2_true_auc:
            if mean_diff > 0:
                power = 1
                outcome = f"CORRECT DETECTION (p < {alpha}, favors {c1_id})"
            else:
                sign_error = 1
                outcome = f"SIGN ERROR (p < {alpha}, favors {c2_id})"
        else:
            if mean_diff < 0:
                power = 1
                outcome = f"CORRECT DETECTION (p < {alpha}, favors {c2_id})"
            else:
                sign_error = 1
                outcome = f"SIGN ERROR (p < {alpha}, favors {c1_id})"
    else:
        outcome = f"NO DETECTION (p >= {alpha})"

    return {
        "power": power,
        "sign_error": sign_error,
        "type1_error": type1_error,
        "mean_difference": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "outcome": outcome,
    }


def evaluate_sample_size(
    sub_results_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    n_sub: int,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all replicates for a specific sample size across both Version A (truth at n)
    and Version B (asymptotic truth).
    """
    dataset_id = str(sub_results_df["dataset_id"].iloc[0])
    truth_A = get_truth_subset(truth_df, dataset_id, n_sub=n_sub, mode="per_sample")
    truth_B = get_truth_subset(truth_df, dataset_id, mode="asymptotic")

    if pairs is None:
        pairs = [("gb", "knn"), ("logreg", "dtree"), ("svm", "logreg")]

    replicate_groups = sub_results_df.groupby("replicate_id")

    ver_A_wc, ver_B_wc = [], []
    ver_A_kt, ver_B_kt = [], []
    ver_A_opt, ver_B_opt = [], []

    pair_results = {p: {"power_A": [], "sign_error_A": [], "power_B": [], "sign_error_B": []} for p in pairs}

    valid_replicates = 0
    invalid_replicates = 0

    for rep_id, rep_df in replicate_groups:
        is_valid, _ = validate_replicate(rep_df, truth_df, n_sub=n_sub)
        if not is_valid:
            invalid_replicates += 1
            continue
        valid_replicates += 1

        rep_summary = calculate_replicate_means(rep_df)

        # Version A
        wc_a = compute_winner_correct(rep_summary, truth_A)
        kt_a = compute_kendall_tau(rep_summary, truth_A)
        opt_a = compute_optimism(rep_summary, truth_A)

        # Version B
        wc_b = compute_winner_correct(rep_summary, truth_B)
        kt_b = compute_kendall_tau(rep_summary, truth_B)
        opt_b = compute_optimism(rep_summary, truth_B)

        ver_A_wc.append(wc_a["is_correct"])
        ver_B_wc.append(wc_b["is_correct"])
        ver_A_kt.append(kt_a["tau"])
        ver_B_kt.append(kt_b["tau"])
        ver_A_opt.append(opt_a["optimism"])
        ver_B_opt.append(opt_b["optimism"])

        # Paired t-tests for each pair
        for (c1, c2) in pairs:
            pt_a = compute_paired_ttest(rep_df, truth_A, c1, c2)
            pt_b = compute_paired_ttest(rep_df, truth_B, c1, c2)
            pair_results[(c1, c2)]["power_A"].append(pt_a["power"])
            pair_results[(c1, c2)]["sign_error_A"].append(pt_a["sign_error"])
            pair_results[(c1, c2)]["power_B"].append(pt_b["power"])
            pair_results[(c1, c2)]["sign_error_B"].append(pt_b["sign_error"])

    pair_summaries = {}
    for p, stats_dict in pair_results.items():
        pair_summaries[p] = {
            "power_A": float(np.mean(stats_dict["power_A"])),
            "sign_error_A": float(np.mean(stats_dict["sign_error_A"])),
            "power_B": float(np.mean(stats_dict["power_B"])),
            "sign_error_B": float(np.mean(stats_dict["sign_error_B"])),
        }

    return {
        "n_sub": n_sub,
        "valid_replicates": valid_replicates,
        "invalid_replicates": invalid_replicates,
        "winner_correct_A": float(np.mean(ver_A_wc)),
        "winner_correct_B": float(np.mean(ver_B_wc)),
        "kendall_tau_A": float(np.mean(ver_A_kt)),
        "kendall_tau_B": float(np.mean(ver_B_kt)),
        "optimism_A": float(np.mean(ver_A_opt)),
        "optimism_B": float(np.mean(ver_B_opt)),
        "pair_summaries": pair_summaries,
    }


def evaluate_experiment(
    truth_source: Union[str, pd.DataFrame] = "reference_truth.csv",
    results_source: Union[str, pd.DataFrame] = "reference_results.csv",
    sample_sizes: Optional[List[int]] = None,
    pairs: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete Week 2 analysis across all sample sizes.
    Generates comparison tables for Version A and Version B.
    """
    truth_df, results_df = load_datasets(truth_source, results_source)

    if sample_sizes is None:
        sample_sizes = sorted([int(x) for x in results_df["n_sub"].dropna().unique()])

    if pairs is None:
        pairs = [
            ("gb", "knn"),        # Close pair (< 0.01 gap at n=50)
            ("svm", "logreg"),    # Close pair (< 0.01 gap at n=200, lead reversal)
            ("logreg", "dtree"),  # Wide pair (> 0.05 gap at all n)
            ("logreg", "knn"),    # Wide pair (> 0.05 gap at all n)
        ]

    results_by_n = {}
    for n in sample_sizes:
        sub_df = results_df[results_df["n_sub"] == n]
        if sub_df.empty:
            continue
        res_n = evaluate_sample_size(sub_df, truth_df, n, pairs=pairs)
        results_by_n[n] = res_n

    if verbose:
        print_full_experiment_report(truth_df, results_by_n, pairs)

    return {
        "results_by_n": results_by_n,
        "truth_df": truth_df,
        "results_df": results_df,
    }


def print_full_experiment_report(
    truth_df: pd.DataFrame,
    results_by_n: Dict[int, Dict[str, Any]],
    pairs: List[Tuple[str, str]],
):
    """Print complete structured evaluation report in high-clarity tables."""
    print("=" * 82)
    print("COMPONENT C (GROUP 5) — WEEK 2 EXPERIMENTAL EVALUATION REPORT")
    print("=" * 82)

    print("\n--- GROUND TRUTH BENCHMARK SUMMARY (reference_truth.csv) ---")
    print(truth_df.to_string(index=False))

    print("\n" + "=" * 82)
    print("TABLE 1: FOUR QUANTITIES × FOUR SAMPLE SIZES (VERSION A vs VERSION B)")
    print("=" * 82)
    header = (
        f"{'Sample Size (n)':<16} | "
        f"{'Winner-Correct (A/B)':<22} | "
        f"{'Kendall Tau (A/B)':<20} | "
        f"{'Optimism (A/B)':<20}"
    )
    print(header)
    print("-" * 82)
    for n, res in results_by_n.items():
        wc_str = f"{res['winner_correct_A']:.4f} / {res['winner_correct_B']:.4f}"
        kt_str = f"{res['kendall_tau_A']:+.4f} / {res['kendall_tau_B']:+.4f}"
        opt_str = f"{res['optimism_A']:+.4f} / {res['optimism_B']:+.4f}"
        print(f"n = {n:<12} | {wc_str:<22} | {kt_str:<20} | {opt_str:<20}")
    print("=" * 82)

    print("\n" + "=" * 82)
    print("TABLE 2: PAIRED t-TEST POWER ANALYSIS (CLOSE PAIR vs WIDE PAIR)")
    print("=" * 82)
    for pair in pairs:
        c1, c2 = pair
        print(f"\n[PAIR: {c1} vs {c2}]")
        print(f"{'Sample Size':<12} | {'True Gap at n':<14} | {'Power (Ver A)':<14} | {'Sign-Error (A)':<14} | {'Power (Ver B)':<14} | {'Sign-Error (B)':<14}")
        print("-" * 92)
        for n, res in results_by_n.items():
            t_sub = get_truth_subset(truth_df, "synthetic_pilot", n_sub=n, mode="per_sample").set_index("classifier_id")["true_auc"].to_dict()
            gap = t_sub[c1] - t_sub[c2]
            p_stats = res["pair_summaries"][pair]
            print(
                f"n = {n:<8} | {gap:+.6f}     | {p_stats['power_A']:<14.4f} | {p_stats['sign_error_A']:<14.4f} | {p_stats['power_B']:<14.4f} | {p_stats['sign_error_B']:<14.4f}"
            )
    print("=" * 92)


def run_unit_tests():
    """
    Comprehensive automated unit test suite for Week 2:
    1. Reference experiment execution (12,000 rows across 4 sample sizes).
    2. Version A vs Version B Optimism behavior (Ver A positive & shrinking; Ver B negative).
    3. Version A vs Version B Winner-Correct behavior (Ver A improves; Ver B low & stagnates).
    4. Version A vs Version B Kendall's Tau behavior.
    5. Rule 1: Empirical mean tie-breaking (lexicographical).
    6. Rules 2 & 5: NaN / non-computable fold AUC rejection.
    7. Rule 3: Duplicate fold IDs rejection.
    8. Rule 3: Shifted fold IDs (e.g. 6..10) rejection.
    9. Rule 3: Non-integer fold IDs (e.g. 1.5) rejection.
    10. Rule 3: Incomplete replicate (fewer than 5 folds) rejection.
    11. Rule 4: Missing true_auc in truth.csv.
    12. Rule 4: Extra classifier in results with no truth entry.
    13. Rule 6: Degenerate paired t-test (zero variance in differences).
    14. Rule 7: Strict p < alpha threshold (p >= alpha boundary and exact p == alpha equality yield NO DETECTION).
    15. Rule 8: Mean paired difference == 0 classified as NO DETECTION.
    16. Truth-Tie Rule: Winner-correct credits any tied true winner.
    17. Truth-Tie Rule: Paired t-test rejections are Type I errors, not power.
    18. Task 3 Power: Close gap (< 0.01) vs Wide gap (> 0.05) power discrimination.
    """
    print("\n" + "#" * 82)
    print("RUNNING COMPONENT C AUTOMATED VERIFICATION TESTS (WEEK 2)")
    print("#" * 82)

    test_results = []

    truth_df, results_df = load_datasets("reference_truth.csv", "reference_results.csv")

    # ==========================================================
    # TEST 1: Full Reference Dataset Execution (12,000 rows)
    # ==========================================================
    print("\n[TEST 1] Testing Reference Dataset Execution across All 4 Sample Sizes...")
    exp_res = evaluate_experiment(truth_df, results_df, sample_sizes=[50, 100, 200, 500], verbose=False)
    res_by_n = exp_res["results_by_n"]
    t1_pass = len(res_by_n) == 4 and all(res_by_n[n]["valid_replicates"] == 100 for n in [50, 100, 200, 500])
    print(f"  - 4 Sample sizes, 100 valid replicates each -> {'PASS' if t1_pass else 'FAIL'}")
    test_results.append(("1. Full 12,000-Row Reference Experiment Execution", t1_pass))

    # ==========================================================
    # TEST 2: Optimism Properties (Theory Verification)
    # ==========================================================
    print("\n[TEST 2] Verifying Optimism Properties (Version A vs Version B)...")
    opt_A_vals = [res_by_n[n]["optimism_A"] for n in [50, 100, 200, 500]]
    opt_B_vals = [res_by_n[n]["optimism_B"] for n in [50, 100, 200, 500]]
    # Ver A must be strictly positive and strictly decreasing
    ver_a_positive = all(x > 0 for x in opt_A_vals)
    ver_a_decreasing = opt_A_vals[0] > opt_A_vals[1] > opt_A_vals[2] > opt_A_vals[3]
    # Ver B must be negative at n=50 (the negative value -0.035 reported by prof)
    ver_b_neg_50 = opt_B_vals[0] < 0
    t2_pass = ver_a_positive and ver_a_decreasing and ver_b_neg_50
    print(f"  - Version A Optimism strictly positive: {ver_a_positive} ({[round(x, 4) for x in opt_A_vals]})")
    print(f"  - Version A Optimism strictly shrinking: {ver_a_decreasing}")
    print(f"  - Version B Optimism at n=50 is negative: {ver_b_neg_50} ({opt_B_vals[0]:.4f}) -> {'PASS' if t2_pass else 'FAIL'}")
    test_results.append(("2. Optimism Properties (Ver A Positive/Shrinking vs Ver B Negative)", t2_pass))

    # ==========================================================
    # TEST 3: Winner-Correct Rate Properties
    # ==========================================================
    print("\n[TEST 3] Verifying Winner-Correct Rate (Version A vs Version B)...")
    wc_A_vals = [res_by_n[n]["winner_correct_A"] for n in [50, 100, 200, 500]]
    wc_B_vals = [res_by_n[n]["winner_correct_B"] for n in [50, 100, 200, 500]]
    # Ver A must be substantially higher than Ver B and increase to high rate at n=500
    t3_pass = (wc_A_vals[-1] > 0.60) and all(a > b for a, b in zip(wc_A_vals, wc_B_vals))
    print(f"  - Version A Winner-Correct: {[round(x, 4) for x in wc_A_vals]}")
    print(f"  - Version B Winner-Correct: {[round(x, 4) for x in wc_B_vals]} -> {'PASS' if t3_pass else 'FAIL'}")
    test_results.append(("3. Winner-Correct Rate Superiority & Convergence under Version A", t3_pass))

    # ==========================================================
    # TEST 4: Kendall's Tau Properties
    # ==========================================================
    print("\n[TEST 4] Verifying Kendall's Tau Rank Correlation...")
    kt_A_vals = [res_by_n[n]["kendall_tau_A"] for n in [50, 100, 200, 500]]
    # Kendall tau under Version A must increase monotonically as sample size grows
    kt_increasing = kt_A_vals[0] < kt_A_vals[1] < kt_A_vals[2] < kt_A_vals[3]
    print(f"  - Version A Kendall's Tau: {[round(x, 4) for x in kt_A_vals]} (Monotonic Increase: {kt_increasing}) -> {'PASS' if kt_increasing else 'FAIL'}")
    test_results.append(("4. Kendall's Tau Monotonic Rank Convergence under Version A", kt_increasing))

    # ==========================================================
    # TEST 5: Rule 1 - Lexicographical Tie Break
    # ==========================================================
    print("\n[TEST 5] Testing Rule 1 (Lexicographical Tie Break on Tied Mean AUC)...")
    tie_summary = pd.DataFrame({
        "dataset_id": ["test", "test"],
        "classifier_id": ["svm", "logreg"],
        "mean_auc": [0.820, 0.820],
    })
    win_clf, win_auc = get_declared_winner(tie_summary)
    t5_pass = (win_clf == "logreg") and (win_auc == 0.820)
    print(f"  - Tied 'svm' & 'logreg' (0.820) -> Declared Winner = '{win_clf}' (Expected 'logreg') -> {'PASS' if t5_pass else 'FAIL'}")
    test_results.append(("5. Rule 1: Lexicographical Tie-Break", t5_pass))

    # ==========================================================
    # TEST 6: Rules 2 & 5 - NaN Fold AUC Rejection
    # ==========================================================
    print("\n[TEST 6] Testing Rules 2 & 5 (Non-computable / NaN Fold AUC Rejection)...")
    sample_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    sample_rep.iloc[0, sample_rep.columns.get_loc("auc")] = np.nan
    is_valid_nan, reason_nan = validate_replicate(sample_rep, truth_df, n_sub=50)
    t6_pass = (is_valid_nan is False) and ("NaN" in reason_nan or "Rules 2 & 5" in reason_nan)
    print(f"  - Replicate with NaN fold -> Valid = {is_valid_nan}, Reason = '{reason_nan}' -> {'PASS' if t6_pass else 'FAIL'}")
    test_results.append(("6. Rules 2 & 5: Non-computable / NaN Fold AUC Rejection", t6_pass))

    # ==========================================================
    # TEST 7: Rule 3 - Duplicate Fold IDs Rejection
    # ==========================================================
    print("\n[TEST 7] Testing Rule 3 (Duplicate Fold IDs in Replicate)...")
    dup_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    c_mask = dup_rep["classifier_id"] == "logreg"
    dup_rep.loc[c_mask, "fold_id"] = [1, 1, 3, 4, 5]
    is_valid_dup, reason_dup = validate_replicate(dup_rep, truth_df, n_sub=50)
    t7_pass = (is_valid_dup is False) and ("invalid fold IDs" in reason_dup or "Rule 3" in reason_dup)
    print(f"  - Duplicate fold_id=1 for logreg -> Valid = {is_valid_dup} -> {'PASS' if t7_pass else 'FAIL'}")
    test_results.append(("7. Rule 3: Duplicate Fold IDs Rejection", t7_pass))

    # ==========================================================
    # TEST 8: Rule 3 - Shifted Fold IDs Range (6..10) Rejection
    # ==========================================================
    print("\n[TEST 8] Testing Rule 3 (Shifted Fold IDs Range: 6..10)...")
    shift_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    c_mask = shift_rep["classifier_id"] == "svm"
    shift_rep.loc[c_mask, "fold_id"] = [6, 7, 8, 9, 10]
    is_valid_shift, reason_shift = validate_replicate(shift_rep, truth_df, n_sub=50)
    t8_pass = (is_valid_shift is False) and ("invalid fold IDs" in reason_shift or "Rule 3" in reason_shift)
    print(f"  - Fold IDs [6..10] -> Valid = {is_valid_shift} -> {'PASS' if t8_pass else 'FAIL'}")
    test_results.append(("8. Rule 3: Shifted Fold IDs Range Rejection", t8_pass))

    # ==========================================================
    # TEST 9: Rule 3 - Non-Integer Fold IDs (e.g. 1.5)
    # ==========================================================
    print("\n[TEST 9] Testing Rule 3 (Non-Integer Fold IDs)...")
    non_int_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    non_int_rep["fold_id"] = non_int_rep["fold_id"].astype(float)
    non_int_rep.iloc[0, non_int_rep.columns.get_loc("fold_id")] = 1.5
    is_valid_ni, reason_ni = validate_replicate(non_int_rep, truth_df, n_sub=50)
    t9_pass = (is_valid_ni is False) and ("non-integer fold IDs" in reason_ni)
    print(f"  - Fold ID = 1.5 -> Valid = {is_valid_ni} -> {'PASS' if t9_pass else 'FAIL'}")
    test_results.append(("9. Rule 3: Non-Integer Fold IDs Rejection", t9_pass))

    # ==========================================================
    # TEST 10: Rule 3 - Incomplete Replicate (Fewer than 5 Folds)
    # ==========================================================
    print("\n[TEST 10] Testing Rule 3 (Fewer than 5 Folds)...")
    incomp_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].iloc[:-1].copy()
    is_valid_inc, reason_inc = validate_replicate(incomp_rep, truth_df, n_sub=50)
    t10_pass = (is_valid_inc is False) and ("Rule 3" in reason_inc)
    print(f"  - Incomplete replicate -> Valid = {is_valid_inc} -> {'PASS' if t10_pass else 'FAIL'}")
    test_results.append(("10. Rule 3: Incomplete Replicate Rejection", t10_pass))

    # ==========================================================
    # TEST 11: Rule 4 - Missing Ground Truth Entry
    # ==========================================================
    print("\n[TEST 11] Testing Rule 4 (Missing Ground Truth Entry)...")
    truth_missing = truth_df[truth_df["classifier_id"] != "gb"].copy()
    sample_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    is_valid_miss, reason_miss = validate_replicate(sample_rep, truth_missing, n_sub=50)
    t11_pass = (is_valid_miss is False) and ("Rule 4" in reason_miss or "missing" in reason_miss)
    print(f"  - Missing 'gb' in truth -> Valid = {is_valid_miss} -> {'PASS' if t11_pass else 'FAIL'}")
    test_results.append(("11. Rule 4: Missing Ground Truth Classifier Rejection", t11_pass))

    # ==========================================================
    # TEST 12: Rule 4 - Extra Classifier with No Ground Truth
    # ==========================================================
    print("\n[TEST 12] Testing Rule 4 (Extra Classifier in Results)...")
    extra_rep = results_df[(results_df["n_sub"] == 50) & (results_df["replicate_id"] == 1)].copy()
    extra_rows = extra_rep[extra_rep["classifier_id"] == "logreg"].copy()
    extra_rows["classifier_id"] = "extra_clf"
    extra_rep = pd.concat([extra_rep, extra_rows], ignore_index=True)
    is_valid_ext, reason_ext = validate_replicate(extra_rep, truth_df, n_sub=50)
    t12_pass = (is_valid_ext is False) and ("Rule 4" in reason_ext or "no matching entry" in reason_ext)
    print(f"  - Extra classifier 'extra_clf' -> Valid = {is_valid_ext} -> {'PASS' if t12_pass else 'FAIL'}")
    test_results.append(("12. Rule 4: Extra Classifier in Results Rejection", t12_pass))

    # ==========================================================
    # TEST 13: Rule 6 - Zero Variance in Differences (Degenerate t-Test)
    # ==========================================================
    print("\n[TEST 13] Testing Rule 6 (Zero Variance in Differences)...")
    rep_zero_var = pd.DataFrame({
        "dataset_id": ["test"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.80, 0.82, 0.78, 0.85, 0.75, 0.70, 0.72, 0.68, 0.75, 0.65],  # Constant difference = +0.10
    })
    truth_zv = pd.DataFrame({
        "dataset_id": ["test", "test"],
        "classifier_id": ["c1", "c2"],
        "true_auc": [0.85, 0.75],
    })
    pw_zv = compute_paired_ttest(rep_zero_var, truth_zv, "c1", "c2")
    t13_pass = (pw_zv["power"] == 0) and ("Zero variance" in pw_zv["outcome"])
    print(f"  - Constant fold difference (zero variance) -> Power = {pw_zv['power']}, Outcome = '{pw_zv['outcome']}' -> {'PASS' if t13_pass else 'FAIL'}")
    test_results.append(("13. Rule 6: Zero Variance in Differences Degenerate t-Test", t13_pass))

    # ==========================================================
    # TEST 14: Rule 7 - Strict p < 0.05 Threshold (p >= 0.05 & Exact p == 0.05 Equality)
    # ==========================================================
    print("\n[TEST 14] Testing Rule 7 (p >= 0.05 Boundary and Exact p == 0.05 Equality)...")
    from unittest.mock import patch
    from collections import namedtuple
    TtestRes = namedtuple("TtestResult", ["statistic", "pvalue"])

    rep_p05 = pd.DataFrame({
        "dataset_id": ["test"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.830, 0.835, 0.798, 0.845, 0.810, 0.800, 0.800, 0.800, 0.800, 0.800],
    })
    # Check 1: Real numerical differences yielding p = 0.05119 >= 0.05 (just above threshold)
    pw_p05 = compute_paired_ttest(rep_p05, truth_zv, "c1", "c2", alpha=0.05)
    pass_above_threshold = (
        pw_p05["power"] == 0
        and pw_p05["sign_error"] == 0
        and pw_p05["p_value"] >= 0.05
        and "NO DETECTION" in pw_p05["outcome"]
    )
    # Check 2: Direct test at EXACT canonical boundary p_value == 0.05 and alpha == 0.05
    with patch("scipy.stats.ttest_rel", return_value=TtestRes(statistic=2.776445, pvalue=0.05)):
        pw_exact_05 = compute_paired_ttest(rep_p05, truth_zv, "c1", "c2", alpha=0.05)
        pass_exact_05 = (
            pw_exact_05["power"] == 0
            and pw_exact_05["sign_error"] == 0
            and pw_exact_05["p_value"] == 0.05
            and "NO DETECTION" in pw_exact_05["outcome"]
        )
    # Check 3: Arbitrary float boundary equality p == alpha (strict < inequality)
    p_exact = float(pw_p05["p_value"])
    pw_exact = compute_paired_ttest(rep_p05, truth_zv, "c1", "c2", alpha=p_exact)
    pass_exact_boundary = (
        pw_exact["power"] == 0
        and pw_exact["sign_error"] == 0
        and "NO DETECTION" in pw_exact["outcome"]
    )
    t14_pass = pass_above_threshold and pass_exact_05 and pass_exact_boundary
    print(f"  - (a) Just above threshold (p = {pw_p05['p_value']:.5f} >= 0.05): Power = {pw_p05['power']} -> {'PASS' if pass_above_threshold else 'FAIL'}")
    print(f"  - (b) Exact standard threshold (p == 0.05, alpha == 0.05): Power = {pw_exact_05['power']} -> {'PASS' if pass_exact_05 else 'FAIL'}")
    print(f"  - (c) Equality boundary (p == alpha = {p_exact:.5f}): Power = {pw_exact['power']} -> {'PASS' if pass_exact_boundary else 'FAIL'}")
    test_results.append(("14. Rule 7: Strict p < 0.05 Threshold (p >= 0.05 & Exact p == 0.05)", t14_pass))

    # ==========================================================
    # TEST 15: Rule 8 - Mean Paired Difference = 0
    # ==========================================================
    print("\n[TEST 15] Testing Rule 8 (Mean Paired Difference = 0)...")
    rep_zero_mean = pd.DataFrame({
        "dataset_id": ["test"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.80, 0.70, 0.75, 0.85, 0.90, 0.70, 0.80, 0.75, 0.90, 0.85],
    })
    pw_zm = compute_paired_ttest(rep_zero_mean, truth_zv, "c1", "c2")
    t15_pass = (pw_zm["power"] == 0) and ("Mean difference = 0.0" in pw_zm["outcome"])
    print(f"  - Mean paired diff = 0.0 -> Power = {pw_zm['power']}, Outcome = '{pw_zm['outcome']}' -> {'PASS' if t15_pass else 'FAIL'}")
    test_results.append(("15. Rule 8: Mean Paired Difference = 0 Handling", t15_pass))

    # ==========================================================
    # TEST 16: Truth-Tie Rule in Winner-Correct Rate
    # ==========================================================
    print("\n[TEST 16] Testing Truth-Tie Rule in Winner-Correct Rate...")
    truth_tie = pd.DataFrame({
        "classifier_id": ["c1", "c2", "c3"],
        "true_auc": [0.85, 0.85, 0.70],
    })
    rep_c1_win = pd.DataFrame({
        "dataset_id": ["t", "t", "t"],
        "classifier_id": ["c1", "c2", "c3"],
        "mean_auc": [0.86, 0.84, 0.72],
    })
    rep_c2_win = pd.DataFrame({
        "dataset_id": ["t", "t", "t"],
        "classifier_id": ["c1", "c2", "c3"],
        "mean_auc": [0.83, 0.87, 0.72],
    })
    wc_1 = compute_winner_correct(rep_c1_win, truth_tie)
    wc_2 = compute_winner_correct(rep_c2_win, truth_tie)
    t16_pass = (wc_1["is_correct"] == 1) and (wc_2["is_correct"] == 1)
    print(f"  - Tied c1=0.85, c2=0.85 -> c1 wins: correct={wc_1['is_correct']}, c2 wins: correct={wc_2['is_correct']} -> {'PASS' if t16_pass else 'FAIL'}")
    test_results.append(("16. Truth-Tie Rule: Preservation in Winner-Correct Rate", t16_pass))

    # ==========================================================
    # TEST 17: Truth-Tie Rule in Paired t-Test (Type I Error)
    # ==========================================================
    print("\n[TEST 17] Testing Truth-Tie Rule in Paired t-Test (Type I Error under H0)...")
    truth_pair_tie = pd.DataFrame({
        "dataset_id": ["tt", "tt"],
        "classifier_id": ["c1", "c2"],
        "true_auc": [0.80, 0.80],
    })
    rep_sig = pd.DataFrame({
        "dataset_id": ["tt"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.90, 0.92, 0.88, 0.91, 0.89, 0.75, 0.76, 0.74, 0.75, 0.74],
    })
    pw_tt = compute_paired_ttest(rep_sig, truth_pair_tie, "c1", "c2")
    t17_pass = (pw_tt["power"] == 0) and (pw_tt["type1_error"] == 1) and ("TYPE I ERROR" in pw_tt["outcome"])
    print(f"  - Significant test with equal ground truth -> Power={pw_tt['power']}, Type1Error={pw_tt['type1_error']} -> {'PASS' if t17_pass else 'FAIL'}")
    test_results.append(("17. Truth-Tie Paired t-Test Type I Error Verification", t17_pass))

    # ==========================================================
    # TEST 18: Close Pair vs Wide Pair Power Discrimination
    # ==========================================================
    print("\n[TEST 18] Verifying Close Pair vs Wide Pair Power Discrimination...")
    # Wide pair logreg vs dtree at n=500 must have high power (> 0.80)
    # Close pair gb vs knn at n=50 must have low power (< 0.10)
    pw_wide_500 = res_by_n[500]["pair_summaries"][("logreg", "dtree")]["power_A"]
    pw_close_50 = res_by_n[50]["pair_summaries"][("gb", "knn")]["power_A"]
    t18_pass = (pw_wide_500 >= 0.80) and (pw_close_50 <= 0.10)
    print(f"  - Wide pair (logreg vs dtree) at n=500 Power = {pw_wide_500:.4f} (>= 0.80)")
    print(f"  - Close pair (gb vs knn) at n=50 Power = {pw_close_50:.4f} (<= 0.10) -> {'PASS' if t18_pass else 'FAIL'}")
    test_results.append(("18. Close Pair vs Wide Pair Power Discrimination", t18_pass))

    # ==========================================================
    # SUMMARY
    # ==========================================================
    print("\n" + "=" * 82)
    print("AUTOMATED UNIT TEST SUITE RESULTS")
    print("=" * 82)
    all_passed = True
    for name, status in test_results:
        print(f"  {name:<65}: [{'PASS' if status else 'FAIL'}]")
        if not status:
            all_passed = False
    print("=" * 82)

    if all_passed:
        print(f"[SUCCESS: ALL {len(test_results)} TEST SUITES PASSED PERFECTLY!]\n")
    else:
        print("[FAILURE: SOME TESTS FAILED!]")
        sys.exit(1)


if __name__ == "__main__":
    # 1. Run full unit test suite
    run_unit_tests()

    # 2. Run full evaluation on reference files and display report
    evaluate_experiment(
        truth_source="reference_truth.csv",
        results_source="reference_results.csv",
        verbose=True,
    )
