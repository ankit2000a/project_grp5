"""
Component C (Group 5) - Statistics Calculation & Verification
CSIT332 Principles of Machine Learning Semester Project

This module implements the calculation and validation of the four statistical metrics:
1. Winner-correct rate
2. Mean Kendall's tau
3. Mean optimism
4. Power (and Sign-error rate) for named pair c1 vs c2 via paired t-tests

==============================================================================
EXPLICIT SPECIFICATION OF THE 8 PRE-SPECIFIED EDGE-CASE RULES
==============================================================================

Rule 1: EXACT TIE IN MEAN AUC
- If two or more classifiers have exactly the same highest mean AUC, select the
  classifier with the smallest classifier_id in lexicographical order as the declared winner.
  Example: c1 = 0.820, c2 = 0.820 -> Winner is c1.

Rule 2: NON-COMPUTABLE FOLD AUC
- If a fold's AUC cannot be computed, treat that fold as invalid.
- Never replace it with 0 or any artificial value.
- If a classifier has fewer than 5 valid folds in that replicate, the replicate is invalid.

Rule 3: INCOMPLETE REPLICATE & FOLD ID INTEGRITY
- Every required classifier must have exactly 5 valid fold rows.
- Fold IDs must be valid integers matching exactly the set {1, 2, 3, 4, 5}.
- Duplicate fold IDs, non-integer fold IDs (e.g. 1.5), missing folds, or out-of-range
  fold IDs (e.g. 6..10) invalidate the entire replicate.

Rule 4: MISSING TRUE AUC & EXTRA CLASSIFIERS
- Filter truth.csv to the current dataset_id.
- Every classifier appearing in the replicate/results MUST have a corresponding valid,
  numeric true_auc in truth.csv for that exact dataset_id + classifier_id.
- If any classifier in results lacks a ground truth entry, or if any ground truth
  entry has missing/NaN true_auc, the replicate is invalid.

Rule 5: MISSING / NON-NUMERIC / NaN AUC
- If an AUC value in results is missing, NaN, or non-numeric, treat that fold as invalid.
- If this leaves fewer than 5 valid unique folds for any required classifier, exclude the replicate.

Rule 6: PAIRED T-TEST CANNOT BE PERFORMED
- If there are insufficient valid paired folds, mismatched fold IDs, zero variance in differences,
  or if ttest_rel fails / produces NaN / inf, classify the replicate as NO DETECTION (power=0, sign_error=0).

Rule 7: p = 0.05 THRESHOLD
- Detection strictly requires p < 0.05.
- Therefore, p = 0.05 (and p > 0.05) is classified as NO DETECTION.

Rule 8: MEAN PAIRED DIFFERENCE = 0
- If the mean paired difference is exactly zero, neither classifier is favored.
- Therefore, it cannot be classified as a correct detection or a sign error (classified as NO DETECTION).
"""

import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def load_datasets(
    truth_source: Union[str, pd.DataFrame] = "truth.csv",
    results_source: Union[str, pd.DataFrame] = "results.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ground truth and results CSV files or dataframes, cleanly coercing types and trimming whitespace.
    """
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

    # Clean whitespace in string identifiers
    truth_df["dataset_id"] = truth_df["dataset_id"].astype(str).str.strip()
    truth_df["classifier_id"] = truth_df["classifier_id"].astype(str).str.strip()

    results_df["dataset_id"] = results_df["dataset_id"].astype(str).str.strip()
    results_df["classifier_id"] = results_df["classifier_id"].astype(str).str.strip()
    if "protocol_id" in results_df.columns:
        results_df["protocol_id"] = results_df["protocol_id"].astype(str).str.strip()

    # Coerce numeric values (Rules 2 & 5)
    truth_df["true_auc"] = pd.to_numeric(truth_df["true_auc"], errors="coerce")
    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    results_df["fold_id"] = pd.to_numeric(results_df["fold_id"], errors="coerce")

    return truth_df, results_df


def validate_replicate(
    replicate_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    expected_folds: int = 5,
    required_classifiers: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate whether a replicate satisfies all integrity and edge-case rules:
    - Rule 4: Match on dataset_id. EVERY classifier in replicate_df MUST have a matching,
      valid, non-NaN true_auc in truth.csv for this dataset_id.
    - Rule 3: Every required classifier must have EXACTLY `expected_folds` rows.
      Fold IDs must be integers and match the exact set {1, 2, ..., expected_folds}.
      Values like 1.5, duplicate fold IDs, or shifted ranges (6..10) invalidate the replicate.
    - Rule 2 & 5: Non-computable / NaN / missing fold AUCs or fold IDs invalidate the replicate.
    """
    if replicate_df.empty:
        return False, "Replicate dataframe is empty."

    # Check dataset_id consistency within replicate
    dataset_ids = replicate_df["dataset_id"].dropna().unique()
    if len(dataset_ids) != 1:
        return False, f"Multiple or missing dataset_ids in replicate: {dataset_ids}"
    dataset_id = str(dataset_ids[0])

    # Filter truth to current dataset_id
    truth_subset = truth_df[truth_df["dataset_id"] == dataset_id]
    if truth_subset.empty:
        return False, f"Rule 4: No ground truth entries found for dataset_id '{dataset_id}'."

    # RULE 4: Check EVERY classifier appearing in results against truth_subset
    replicate_clfs = set(replicate_df["classifier_id"].dropna().unique())
    for clf in replicate_clfs:
        clf_truth = truth_subset[truth_subset["classifier_id"] == clf]
        if clf_truth.empty:
            return False, (
                f"Rule 4: Classifier '{clf}' in results has no matching entry in truth.csv "
                f"for dataset_id '{dataset_id}'."
            )
        if pd.isna(clf_truth["true_auc"].iloc[0]):
            return False, (
                f"Rule 4: Classifier '{clf}' in truth.csv has missing/NaN true_auc "
                f"for dataset_id '{dataset_id}'."
            )

    # Determine required classifiers
    if required_classifiers is None:
        required_classifiers = sorted(
            truth_subset["classifier_id"].dropna().unique().tolist()
        )

    # Check that all required classifiers are present in the replicate
    missing_required = set(required_classifiers) - replicate_clfs
    if missing_required:
        return False, f"Rule 3: Replicate is missing required classifiers: {sorted(list(missing_required))}."

    # Expected exact set of integer fold IDs {1, 2, ..., expected_folds}
    expected_fold_ids = set(range(1, expected_folds + 1))

    # RULE 3 & 5: Check fold integrity for each required classifier
    for clf in required_classifiers:
        clf_rows = replicate_df[replicate_df["classifier_id"] == clf]

        # Check total rows count
        if len(clf_rows) != expected_folds:
            return False, (
                f"Rule 3: Classifier '{clf}' has {len(clf_rows)} rows, expected exactly {expected_folds}."
            )

        # Check for NaN / non-numeric AUC or fold_id (Rules 2 & 5)
        if clf_rows["auc"].isna().any():
            return False, f"Rules 2 & 5: Classifier '{clf}' contains NaN/non-computable fold AUC."
        if clf_rows["fold_id"].isna().any():
            return False, f"Rules 2 & 5: Classifier '{clf}' contains NaN/non-numeric fold_id."

        # Check that fold IDs are integer-valued (e.g. not 1.5)
        fold_id_vals = clf_rows["fold_id"].to_numpy(dtype=float)
        if not np.all(np.equal(np.mod(fold_id_vals, 1), 0)):
            return False, f"Rule 3: Classifier '{clf}' has non-integer fold IDs."

        # Check that fold IDs match the exact required set {1, 2, ..., expected_folds}
        actual_fold_ids = set(int(x) for x in fold_id_vals)
        if actual_fold_ids != expected_fold_ids:
            return False, (
                f"Rule 3: Classifier '{clf}' has invalid fold IDs {sorted(list(actual_fold_ids))}, "
                f"expected exact set {sorted(list(expected_fold_ids))}."
            )

    return True, "Valid replicate."


def calculate_replicate_means(replicate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each classifier's replicate-level AUC as the mean of its fold AUCs.
    """
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
    Determine the declared winner based on highest mean observed AUC.
    Rule 1 (Exact tie in mean AUC): Deterministically broken by selecting the
    lexicographically smallest classifier_id (ascending order).
    """
    sorted_df = replicate_summary.sort_values(
        by=["mean_auc", "classifier_id"], ascending=[False, True]
    )
    winner_row = sorted_df.iloc[0]
    return str(winner_row["classifier_id"]), float(winner_row["mean_auc"])


def get_true_best_classifier(
    truth_df: pd.DataFrame, dataset_id: str
) -> Tuple[str, float]:
    """
    Determine the genuinely best classifier from ground truth for the given dataset_id.
    Tie-breaking: Lexicographically smallest classifier_id if true AUCs tie.
    """
    truth_subset = truth_df[truth_df["dataset_id"] == dataset_id]
    sorted_truth = truth_subset.sort_values(
        by=["true_auc", "classifier_id"], ascending=[False, True]
    )
    best_row = sorted_truth.iloc[0]
    return str(best_row["classifier_id"]), float(best_row["true_auc"])


def compute_winner_correct(
    replicate_summary: pd.DataFrame, truth_df: pd.DataFrame, dataset_id: str
) -> Dict[str, Any]:
    """
    1. Winner-correct rate:
    - Declared winner: highest observed replicate mean AUC (Rule 1 applied for ties).
    - True best: highest true_auc from truth.csv for dataset_id.
    - Correct = 1 if declared_winner == true_best else 0.
    """
    declared_winner, declared_mean_auc = get_declared_winner(replicate_summary)
    true_best, true_best_auc = get_true_best_classifier(truth_df, dataset_id)

    is_correct = 1 if declared_winner == true_best else 0
    return {
        "is_correct": is_correct,
        "declared_winner": declared_winner,
        "declared_mean_auc": declared_mean_auc,
        "true_best": true_best,
        "true_best_auc": true_best_auc,
    }


def compute_kendall_tau(
    replicate_summary: pd.DataFrame, truth_df: pd.DataFrame, dataset_id: str
) -> Dict[str, Any]:
    """
    2. Kendall's tau:
    - Filter truth.csv to dataset_id.
    - Match and merge on BOTH dataset_id and classifier_id.
    - Rank classifiers by observed mean AUC vs true_auc using scipy.stats.kendalltau.
    """
    truth_subset = truth_df[truth_df["dataset_id"] == dataset_id]

    # Merge on BOTH dataset_id and classifier_id
    merged = pd.merge(
        replicate_summary, truth_subset, on=["dataset_id", "classifier_id"]
    )

    # Sort deterministically by classifier_id to maintain aligned vector order
    merged = merged.sort_values(by="classifier_id").reset_index(drop=True)

    tau_res = stats.kendalltau(merged["true_auc"], merged["mean_auc"])
    tau_val = float(tau_res.statistic) if not np.isnan(tau_res.statistic) else 0.0

    observed_rank = (
        merged.sort_values(
            by=["mean_auc", "classifier_id"], ascending=[False, True]
        )["classifier_id"]
        .tolist()
    )
    true_rank = (
        merged.sort_values(
            by=["true_auc", "classifier_id"], ascending=[False, True]
        )["classifier_id"]
        .tolist()
    )

    return {
        "tau": tau_val,
        "p_value": float(tau_res.pvalue) if not np.isnan(tau_res.pvalue) else 1.0,
        "observed_ranking": observed_rank,
        "true_ranking": true_rank,
        "comparison_table": merged[
            ["dataset_id", "classifier_id", "true_auc", "mean_auc"]
        ],
    }


def compute_optimism(
    replicate_summary: pd.DataFrame, truth_df: pd.DataFrame, dataset_id: str
) -> Dict[str, Any]:
    """
    3. Optimism:
    - Identify declared winner from observed results.
    - optimism = observed mean AUC of declared winner - true_auc of THAT SAME winner.
    - IMPORTANT: Do NOT subtract the true AUC of the genuinely best classifier.
    """
    declared_winner, declared_mean_auc = get_declared_winner(replicate_summary)
    truth_subset = truth_df[truth_df["dataset_id"] == dataset_id]

    winner_truth_row = truth_subset[truth_subset["classifier_id"] == declared_winner]
    winner_true_auc = float(winner_truth_row["true_auc"].iloc[0])

    optimism = declared_mean_auc - winner_true_auc
    return {
        "declared_winner": declared_winner,
        "declared_mean_auc": declared_mean_auc,
        "winner_true_auc": winner_true_auc,
        "optimism": float(optimism),
    }


def compute_power_c1_vs_c2(
    replicate_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    dataset_id: str,
    c1_id: str = "c1",
    c2_id: str = "c2",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    4. Power for c1 vs c2:
    - Named pair c1 and c2. Ground truth verified: true_auc(c1) > true_auc(c2).
    - Pair fold 1 of c1 with fold 1 of c2, fold 2 with fold 2, etc.
    - Differences: c1 - c2.
    - Rule 6: Insufficient folds, mismatched fold IDs, zero variance, or ttest failure -> NO DETECTION.
    - Rule 7: Detection requires p < 0.05 (p = 0.05 is NO DETECTION).
    - Rule 8: Mean paired difference == 0 -> NO DETECTION.
    - Correct detection: p < alpha AND mean difference > 0 (favors c1).
    - Sign error: p < alpha AND mean difference < 0 (favors c2).
    - No detection: p >= alpha OR mean difference == 0.
    """
    c1_folds = replicate_df[replicate_df["classifier_id"] == c1_id].sort_values(
        by="fold_id"
    )
    c2_folds = replicate_df[replicate_df["classifier_id"] == c2_id].sort_values(
        by="fold_id"
    )

    # Rule 6: Check fold matching and count
    if (
        c1_folds.empty
        or c2_folds.empty
        or len(c1_folds) < 2
        or len(c1_folds) != len(c2_folds)
        or list(c1_folds["fold_id"]) != list(c2_folds["fold_id"])
    ):
        return {
            "c1_auc_folds": c1_folds["auc"].tolist() if not c1_folds.empty else [],
            "c2_auc_folds": c2_folds["auc"].tolist() if not c2_folds.empty else [],
            "paired_diffs": [],
            "mean_difference": 0.0,
            "t_statistic": np.nan,
            "p_value": 1.0,
            "power": 0,
            "sign_error": 0,
            "outcome": "NO DETECTION (Rule 6: Insufficient/unmatched paired folds)",
        }

    c1_vals = c1_folds["auc"].to_numpy(dtype=float)
    c2_vals = c2_folds["auc"].to_numpy(dtype=float)
    diffs = c1_vals - c2_vals
    mean_diff = float(np.mean(diffs))

    # Rule 8: If mean paired difference is exactly zero (neither classifier favored)
    if np.isclose(mean_diff, 0.0, atol=1e-12):
        return {
            "c1_auc_folds": c1_vals.tolist(),
            "c2_auc_folds": c2_vals.tolist(),
            "paired_diffs": diffs.tolist(),
            "mean_difference": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "power": 0,
            "sign_error": 0,
            "outcome": "NO DETECTION (Rule 8: Mean difference = 0.0)",
        }

    # Rule 6: Check for zero variance in differences (degenerate t-test)
    diff_std = float(np.std(diffs, ddof=1))
    if np.isclose(diff_std, 0.0, atol=1e-12):
        return {
            "c1_auc_folds": c1_vals.tolist(),
            "c2_auc_folds": c2_vals.tolist(),
            "paired_diffs": diffs.tolist(),
            "mean_difference": mean_diff,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "power": 0,
            "sign_error": 0,
            "outcome": "NO DETECTION (Rule 6: Degenerate paired t-test / zero variance)",
        }

    # Perform paired t-test
    try:
        ttest_res = stats.ttest_rel(c1_vals, c2_vals)
        t_stat = float(ttest_res.statistic)
        p_val = float(ttest_res.pvalue)
    except Exception:
        t_stat, p_val = np.nan, 1.0

    # Rule 6: Handle NaN/inf t-statistic or p-value
    if np.isnan(p_val) or np.isnan(t_stat) or np.isinf(t_stat):
        return {
            "c1_auc_folds": c1_vals.tolist(),
            "c2_auc_folds": c2_vals.tolist(),
            "paired_diffs": diffs.tolist(),
            "mean_difference": mean_diff,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "power": 0,
            "sign_error": 0,
            "outcome": "NO DETECTION (Rule 6: Degenerate t-test)",
        }

    # Rule 7: Detection strictly requires p < alpha (p = 0.05 is NO DETECTION)
    if p_val < alpha and mean_diff > 0:
        power = 1
        sign_error = 0
        outcome = "CORRECT DETECTION (p < 0.05, favors c1)"
    elif p_val < alpha and mean_diff < 0:
        power = 0
        sign_error = 1
        outcome = "SIGN ERROR (p < 0.05, favors c2)"
    else:
        power = 0
        sign_error = 0
        outcome = "NO DETECTION (p >= 0.05)"

    return {
        "c1_auc_folds": c1_vals.tolist(),
        "c2_auc_folds": c2_vals.tolist(),
        "paired_diffs": diffs.tolist(),
        "mean_difference": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "power": power,
        "sign_error": sign_error,
        "outcome": outcome,
    }


def evaluate_dataset(
    truth_source: Union[str, pd.DataFrame] = "truth.csv",
    results_source: Union[str, pd.DataFrame] = "results.csv",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all replicates in results_source against truth_source.
    Calculates the 4 metrics averaged across all valid replicates in the input.
    """
    truth_df, results_df = load_datasets(truth_source, results_source)

    group_cols = ["dataset_id", "replicate_id"]
    if "protocol_id" in results_df.columns:
        group_cols.insert(1, "protocol_id")
    if "n_sub" in results_df.columns:
        group_cols.insert(2, "n_sub")

    grouped = results_df.groupby(group_cols)

    replicate_records = []
    invalid_replicates = []

    for group_keys, rep_df in grouped:
        if isinstance(group_keys, tuple):
            dataset_id = str(group_keys[0])
            rep_id = group_keys[-1]
        else:
            dataset_id = str(rep_df["dataset_id"].iloc[0])
            rep_id = group_keys

        # Check replicate validity (Rules 2, 3, 4, 5)
        is_valid, reason = validate_replicate(rep_df, truth_df)
        if not is_valid:
            invalid_replicates.append((group_keys, reason))
            continue

        rep_summary = calculate_replicate_means(rep_df)

        wc = compute_winner_correct(rep_summary, truth_df, dataset_id)
        kt = compute_kendall_tau(rep_summary, truth_df, dataset_id)
        opt = compute_optimism(rep_summary, truth_df, dataset_id)
        pw = compute_power_c1_vs_c2(rep_df, truth_df, dataset_id)

        rec = {
            "group_keys": group_keys,
            "dataset_id": dataset_id,
            "replicate_id": rep_id,
            "summary": rep_summary,
            "winner_correct": wc,
            "kendall_tau": kt,
            "optimism": opt,
            "power_c1_c2": pw,
        }
        replicate_records.append(rec)

    if not replicate_records:
        raise ValueError(
            f"No valid replicates found. Excluded reasons: {invalid_replicates}"
        )

    winner_correct_rate = float(
        np.mean([r["winner_correct"]["is_correct"] for r in replicate_records])
    )
    mean_kendall_tau = float(
        np.mean([r["kendall_tau"]["tau"] for r in replicate_records])
    )
    mean_optimism = float(
        np.mean([r["optimism"]["optimism"] for r in replicate_records])
    )
    power_rate = float(
        np.mean([r["power_c1_c2"]["power"] for r in replicate_records])
    )
    sign_error_rate = float(
        np.mean([r["power_c1_c2"]["sign_error"] for r in replicate_records])
    )

    results = {
        "valid_replicates_count": len(replicate_records),
        "invalid_replicates_count": len(invalid_replicates),
        "winner_correct_rate": winner_correct_rate,
        "mean_kendall_tau": mean_kendall_tau,
        "mean_optimism": mean_optimism,
        "power": power_rate,
        "sign_error_rate": sign_error_rate,
        "replicates": replicate_records,
    }

    if verbose:
        print_detailed_report(truth_df, results_df, results)

    return results


def evaluate_by_sample_size(
    truth_path: str = "truth.csv",
    results_path: str = "results.csv",
    sample_sizes: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate statistics separately for each sample size (n_sub).
    Ensures sample sizes are not averaged together.
    """
    if sample_sizes is None:
        sample_sizes = [100, 500]

    truth_df, results_df = load_datasets(truth_path, results_path)

    results_by_n = {}

    for n_sub in sample_sizes:
        sub_df = results_df[results_df["n_sub"] == n_sub].copy()
        if sub_df.empty:
            continue

        eval_n = evaluate_dataset(truth_df, sub_df, verbose=False)
        results_by_n[n_sub] = eval_n

    if verbose:
        print("\n" + "=" * 78)
        print("FINAL STATISTICAL SUMMARY BY SAMPLE SIZE")
        print("=" * 78)

        for n_sub in sample_sizes:
            if n_sub not in results_by_n:
                continue
            res = results_by_n[n_sub]
            print(f"\n[SAMPLE SIZE: n_sub = {n_sub}]")
            print("-" * 45)
            print(f"  Valid Replicates Analyzed   : {res['valid_replicates_count']}")
            print(f"  Invalid Replicates Excluded : {res['invalid_replicates_count']}")
            print(f"  1. Winner-correct rate      : {res['winner_correct_rate']:.4f}")
            print(f"  2. Mean Kendall's tau       : {res['mean_kendall_tau']:.6f}")
            print(f"  3. Mean optimism            : {res['mean_optimism']:.6f}")
            print(f"  4. Power (c1 vs c2)         : {res['power']:.4f}")
            print(f"     Sign-error rate          : {res['sign_error_rate']:.4f}")
            print("-" * 45)

        print("=" * 78)

    return results_by_n


def print_detailed_report(
    truth_df: pd.DataFrame, results_df: pd.DataFrame, results: Dict[str, Any]
):
    """Print complete intermediate calculations, tables, and statistics summary."""
    print("=" * 78)
    print("COMPONENT C — DETAILED EVALUATION REPORT")
    print("=" * 78)

    print("\n--- 1. GROUND TRUTH (truth.csv) ---")
    print(truth_df.to_string(index=False))

    print("\n--- 2. OBSERVED RESULTS ---")
    if len(results_df) <= 20:
        print(results_df.to_string(index=False))
    else:
        print(f"(Showing summary for {len(results_df)} total rows across {results['valid_replicates_count']} valid replicates)")

    for idx, rep in enumerate(results["replicates"], 1):
        print("\n" + "=" * 78)
        print(
            f"REPLICATE {rep['replicate_id']} (Dataset: {rep['dataset_id']}) — INTERMEDIATE CALCULATIONS"
        )
        print("=" * 78)

        print("\n[A] Fold AUC Values & Replicate-level Means:")
        for _, row in rep["summary"].iterrows():
            folds_formatted = [f"{x:.4f}" for x in row["fold_aucs"]]
            print(
                f"  Classifier {row['classifier_id']:<3}: Folds = [{', '.join(folds_formatted)}] "
                f"-> Mean AUC = {row['mean_auc']:.4f}"
            )

        wc = rep["winner_correct"]
        print("\n[B] Statistic 1: Winner-Correct Rate")
        print(
            f"  Observed Winner : {wc['declared_winner']} (Mean AUC = {wc['declared_mean_auc']:.4f})"
        )
        print(
            f"  True Winner     : {wc['true_best']} (True AUC = {wc['true_best_auc']:.4f})"
        )
        print(f"  Is Correct?     : {bool(wc['is_correct'])}")
        print(f"  >> Winner-Correct Score = {wc['is_correct']}")

        kt = rep["kendall_tau"]
        print("\n[C] Statistic 2: Kendall's Tau")
        print(f"  Observed Ranking : {' > '.join(kt['observed_ranking'])}")
        print(f"  True Ranking     : {' > '.join(kt['true_ranking'])}")
        print("  Classifier Comparison Table (Matched on dataset_id & classifier_id):")
        print(
            "    "
            + kt["comparison_table"].to_string(index=False).replace("\n", "\n    ")
        )
        print(
            f"  scipy.stats.kendalltau statistic = {kt['tau']:.10f} (p-value = {kt['p_value']:.4f})"
        )
        print(f"  >> Kendall's tau = {kt['tau']:.10f} (Exact 1/3 = {1/3:.10f})")

        opt = rep["optimism"]
        print("\n[D] Statistic 3: Mean Optimism")
        print(f"  Declared Winner             : {opt['declared_winner']}")
        print(f"  Observed Mean AUC of Winner : {opt['declared_mean_auc']:.4f}")
        print(f"  True AUC of Winner          : {opt['winner_true_auc']:.4f}")
        print(
            f"  Optimism Formula            : {opt['declared_mean_auc']:.4f} - {opt['winner_true_auc']:.4f} "
            f"= {opt['optimism']:.4f}"
        )
        print(f"  >> Optimism = {opt['optimism']:.4f}")

        pw = rep["power_c1_c2"]
        print("\n[E] Statistic 4: Power & Sign Error (c1 vs c2)")
        print(f"  c1 Fold AUCs        : {[round(x, 4) for x in pw['c1_auc_folds']]}")
        print(f"  c2 Fold AUCs        : {[round(x, 4) for x in pw['c2_auc_folds']]}")
        print(f"  Paired Diffs (c1-c2): {[round(x, 4) for x in pw['paired_diffs']]}")
        print(f"  Mean Difference     : {pw['mean_difference']:.4f}")
        print(
            f"  Paired t-test       : t = {pw['t_statistic']:.4f}, p = {pw['p_value']:.4f}"
        )
        print(f"  Classification      : {pw['outcome']}")
        print(f"  >> Power (Detection) = {pw['power']}")
        print(f"  >> Sign Error        = {pw['sign_error']}")

    print("\n" + "=" * 78)
    print("SUMMARY OF THE FOUR STATISTICS")
    print("=" * 78)
    print(f"  Valid Replicates Analyzed : {results['valid_replicates_count']}")
    print(f"  1. Winner-correct rate    : {results['winner_correct_rate']}")
    print(f"  2. Mean Kendall's tau     : {results['mean_kendall_tau']:.10f}")
    print(f"  3. Mean optimism          : {results['mean_optimism']:.6f}")
    print(f"  4. Power (c1 vs c2)       : {results['power']}")
    print(f"     Sign-error rate        : {results['sign_error_rate']}")
    print("=" * 78)


def run_unit_tests():
    """
    Automated test suite asserting:
    1. Main toy dataset reproduces exact expected answers on toy_results.csv.
    2. Rule 1: Exact mean-AUC tie-breaking (lexicographically smallest classifier_id).
    3. Rules 2 & 5: Missing / non-numeric / NaN fold AUC.
    4. Rule 3: Duplicate fold IDs in replicate (e.g. two fold 1s).
    5. Rule 3: Invalid / shifted fold IDs range (e.g. 6..10 instead of 1..5).
    6. Rule 3: Non-integer fold IDs (e.g. 1.5).
    7. Rule 3: Incomplete replicate (e.g. 4 folds instead of 5).
    8. Rule 4: Missing true_auc in truth.csv for a classifier.
    9. Rule 4: Extra classifier in results with no truth entry.
    10. Rule 6: Degenerate paired t-test (zero variance in differences).
    11. Rule 7: p >= 0.05 boundary is classified as NO DETECTION.
    12. Rule 8: Mean paired difference = 0 classified as NO DETECTION.
    13. Sample Size Separation: evaluate_by_sample_size independently evaluates n_sub=100 and n_sub=500.
    """
    print("\n" + "#" * 78)
    print("STARTING AUTOMATED VERIFICATION TESTS (COMPONENT C)")
    print("#" * 78)

    test_results = []

    # ==========================================================
    # TEST 1: Main Toy Dataset Verification (using toy_results.csv)
    # ==========================================================
    print("\n[TEST 1] Testing Toy Dataset against Hand-Calculated Values (toy_results.csv)...")
    toy_eval = evaluate_dataset("truth.csv", "toy_results.csv", verbose=False)

    expected_wc = 0
    expected_kt = 1.0 / 3.0
    expected_opt = 0.0100
    expected_pw = 0
    expected_se = 0

    pass_wc = toy_eval["winner_correct_rate"] == expected_wc
    pass_kt = np.isclose(toy_eval["mean_kendall_tau"], expected_kt, atol=1e-6)
    pass_opt = np.isclose(toy_eval["mean_optimism"], expected_opt, atol=1e-6)
    pass_pw = toy_eval["power"] == expected_pw
    pass_se = toy_eval["sign_error_rate"] == expected_se

    test1_passed = all([pass_wc, pass_kt, pass_opt, pass_pw, pass_se])

    print(
        f"  - Winner-correct Rate : Calculated = {toy_eval['winner_correct_rate']}, Expected = {expected_wc} -> {'PASS' if pass_wc else 'FAIL'}"
    )
    print(
        f"  - Mean Kendall's Tau  : Calculated = {toy_eval['mean_kendall_tau']:.6f}, Expected = {expected_kt:.6f} -> {'PASS' if pass_kt else 'FAIL'}"
    )
    print(
        f"  - Mean Optimism       : Calculated = {toy_eval['mean_optimism']:.4f}, Expected = {expected_opt:.4f} -> {'PASS' if pass_opt else 'FAIL'}"
    )
    print(
        f"  - Power (c1 vs c2)    : Calculated = {toy_eval['power']}, Expected = {expected_pw} -> {'PASS' if pass_pw else 'FAIL'}"
    )
    print(
        f"  - Sign Error Rate     : Calculated = {toy_eval['sign_error_rate']}, Expected = {expected_se} -> {'PASS' if pass_se else 'FAIL'}"
    )
    test_results.append(("1. Toy Dataset Verification", test1_passed))

    truth_df, toy_df = load_datasets("truth.csv", "toy_results.csv")

    # ==========================================================
    # TEST 2: Rule 1 - Exact Tie in Mean AUC
    # ==========================================================
    print("\n[TEST 2] Testing Rule 1 (Exact Tie in Mean AUC -> Lexicographical Tie Break)...")
    tie_summary = pd.DataFrame({
        "dataset_id": ["toy", "toy", "toy"],
        "classifier_id": ["c2", "c1", "c3"],
        "mean_auc": [0.820, 0.820, 0.760],
    })
    winner_clf, winner_auc = get_declared_winner(tie_summary)
    rule1_passed = (winner_clf == "c1") and (winner_auc == 0.820)
    print(
        f"  - Tied c1 (0.82) & c2 (0.82) -> Declared Winner = '{winner_clf}' (Expected 'c1') -> {'PASS' if rule1_passed else 'FAIL'}"
    )
    test_results.append(("2. Rule 1: Exact Tie Lexicographical Tie-Break", rule1_passed))

    # ==========================================================
    # TEST 3: Rule 2 & 5 - Non-computable / NaN Fold AUC
    # ==========================================================
    print("\n[TEST 3] Testing Rules 2 & 5 (Non-computable / NaN Fold AUC)...")
    nan_rep = toy_df.copy()
    nan_rep.loc[0, "auc"] = np.nan
    is_valid_nan, reason_nan = validate_replicate(nan_rep, truth_df)
    rule2_5_passed = (is_valid_nan is False) and ("NaN" in reason_nan or "Rules 2 & 5" in reason_nan)
    print(
        f"  - Replicate with NaN fold -> Valid = {is_valid_nan}, Reason = '{reason_nan}' -> {'PASS' if rule2_5_passed else 'FAIL'}"
    )
    test_results.append(("3. Rules 2 & 5: Non-computable / NaN Fold", rule2_5_passed))

    # ==========================================================
    # TEST 4: Rule 3 - Duplicate Fold IDs
    # ==========================================================
    print("\n[TEST 4] Testing Rule 3 (Duplicate Fold IDs in Replicate)...")
    dup_fold_rep = toy_df.copy()
    dup_fold_rep.loc[1, "fold_id"] = 1  # Replace fold 2 of c1 with another fold 1
    is_valid_dup, reason_dup = validate_replicate(dup_fold_rep, truth_df)
    rule3_dup_passed = (is_valid_dup is False) and ("invalid fold IDs" in reason_dup or "Rule 3" in reason_dup)
    print(
        f"  - Replicate with duplicate fold_id=1 -> Valid = {is_valid_dup}, Reason = '{reason_dup}' -> {'PASS' if rule3_dup_passed else 'FAIL'}"
    )
    test_results.append(("4. Rule 3: Duplicate Fold IDs Rejection", rule3_dup_passed))

    # ==========================================================
    # TEST 5: Rule 3 - Invalid Fold IDs Range (6..10 instead of 1..5)
    # ==========================================================
    print("\n[TEST 5] Testing Rule 3 (Invalid Fold IDs Range: 6..10)...")
    shifted_fold_rep = toy_df.copy()
    c1_mask = shifted_fold_rep["classifier_id"] == "c1"
    shifted_fold_rep.loc[c1_mask, "fold_id"] = [6, 7, 8, 9, 10]
    is_valid_shift, reason_shift = validate_replicate(shifted_fold_rep, truth_df)
    rule3_shift_passed = (is_valid_shift is False) and ("Rule 3" in reason_shift and "invalid fold IDs" in reason_shift)
    print(
        f"  - Classifier c1 with fold IDs [6..10] -> Valid = {is_valid_shift}, Reason = '{reason_shift}' -> {'PASS' if rule3_shift_passed else 'FAIL'}"
    )
    test_results.append(("5. Rule 3: Shifted Fold IDs Range Rejection", rule3_shift_passed))

    # ==========================================================
    # TEST 6: Rule 3 - Non-Integer Fold IDs (e.g. 1.5)
    # ==========================================================
    print("\n[TEST 6] Testing Rule 3 (Non-Integer Fold IDs)...")
    non_int_fold_rep = toy_df.copy()
    non_int_fold_rep["fold_id"] = non_int_fold_rep["fold_id"].astype(float)
    non_int_fold_rep.loc[0, "fold_id"] = 1.5
    is_valid_non_int, reason_non_int = validate_replicate(non_int_fold_rep, truth_df)
    rule3_non_int_passed = (is_valid_non_int is False) and ("non-integer fold IDs" in reason_non_int)
    print(
        f"  - Classifier c1 with fold_id=1.5 -> Valid = {is_valid_non_int}, Reason = '{reason_non_int}' -> {'PASS' if rule3_non_int_passed else 'FAIL'}"
    )
    test_results.append(("6. Rule 3: Non-Integer Fold IDs Rejection", rule3_non_int_passed))

    # ==========================================================
    # TEST 7: Rule 3 - Incomplete Replicate (Fewer than 5 Folds)
    # ==========================================================
    print("\n[TEST 7] Testing Rule 3 (Incomplete Replicate: 4 Folds)...")
    inc_rep = toy_df.iloc[:-1].copy()  # drop 1 fold of c3
    is_valid_inc, reason_inc = validate_replicate(inc_rep, truth_df)
    rule3_inc_passed = (is_valid_inc is False) and ("expected exactly 5" in reason_inc)
    print(
        f"  - Classifier c3 with 4 rows -> Valid = {is_valid_inc}, Reason = '{reason_inc}' -> {'PASS' if rule3_inc_passed else 'FAIL'}"
    )
    test_results.append(("7. Rule 3: Incomplete Fold Count Rejection", rule3_inc_passed))

    # ==========================================================
    # TEST 8: Rule 4 - Missing true_auc in truth.csv
    # ==========================================================
    print("\n[TEST 8] Testing Rule 4 (Missing / NaN true_auc in truth.csv)...")
    nan_truth = truth_df.copy()
    nan_truth.loc[0, "true_auc"] = np.nan
    is_valid_t_nan, reason_t_nan = validate_replicate(toy_df, nan_truth)
    rule4_nan_passed = (is_valid_t_nan is False) and ("missing/NaN true_auc" in reason_t_nan)
    print(
        f"  - truth.csv with NaN true_auc -> Valid = {is_valid_t_nan}, Reason = '{reason_t_nan}' -> {'PASS' if rule4_nan_passed else 'FAIL'}"
    )
    test_results.append(("8. Rule 4: Missing true_auc in truth.csv", rule4_nan_passed))

    # ==========================================================
    # TEST 9: Rule 4 - Extra Classifier in Results with No Truth Entry
    # ==========================================================
    print("\n[TEST 9] Testing Rule 4 (Extra Classifier in Results with No Truth Entry)...")
    extra_clf_rep = toy_df.copy()
    extra_row = pd.DataFrame([{
        "dataset_id": "toy",
        "protocol_id": "cv5",
        "n_sub": 100,
        "replicate_id": 1,
        "classifier_id": "c4",
        "fold_id": 1,
        "auc": 0.80,
    }])
    extra_clf_rep = pd.concat([extra_clf_rep, extra_row], ignore_index=True)
    is_valid_extra, reason_extra = validate_replicate(extra_clf_rep, truth_df)
    rule4_extra_passed = (is_valid_extra is False) and ("no matching entry in truth.csv" in reason_extra)
    print(
        f"  - Results with unrecorded 'c4' -> Valid = {is_valid_extra}, Reason = '{reason_extra}' -> {'PASS' if rule4_extra_passed else 'FAIL'}"
    )
    test_results.append(("9. Rule 4: Extra Classifier with No Truth Entry", rule4_extra_passed))

    # ==========================================================
    # TEST 10: Rule 6 - Degenerate Paired t-test / Zero Variance
    # ==========================================================
    print("\n[TEST 10] Testing Rule 6 (Degenerate Paired t-test / Zero Variance)...")
    deg_rep = pd.DataFrame({
        "dataset_id": ["toy"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.85, 0.85, 0.85, 0.85, 0.85, 0.80, 0.80, 0.80, 0.80, 0.80],
    })
    pw_deg = compute_power_c1_vs_c2(deg_rep, truth_df, "toy")
    rule6_passed = (pw_deg["power"] == 0 and pw_deg["sign_error"] == 0 and "Rule 6" in pw_deg["outcome"])
    print(
        f"  - Zero variance diffs -> Outcome = '{pw_deg['outcome']}', Power = {pw_deg['power']} -> {'PASS' if rule6_passed else 'FAIL'}"
    )
    test_results.append(("10. Rule 6: Degenerate Paired t-test", rule6_passed))

    # ==========================================================
    # TEST 11: Rule 7 - p = 0.05 Threshold Boundary
    # ==========================================================
    print("\n[TEST 11] Testing Rule 7 (p >= 0.05 is NO DETECTION)...")
    pw_toy = compute_power_c1_vs_c2(toy_df, truth_df, "toy")
    rule7_passed = (pw_toy["p_value"] >= 0.05 and pw_toy["power"] == 0 and pw_toy["sign_error"] == 0)
    print(
        f"  - p = {pw_toy['p_value']:.4f} (>= 0.05) -> Power = {pw_toy['power']}, Sign Error = {pw_toy['sign_error']} -> {'PASS' if rule7_passed else 'FAIL'}"
    )
    test_results.append(("11. Rule 7: p >= 0.05 Boundary Classification", rule7_passed))

    # ==========================================================
    # TEST 12: Rule 8 - Mean Paired Difference = 0
    # ==========================================================
    print("\n[TEST 12] Testing Rule 8 (Mean Paired Difference = 0)...")
    zero_diff_rep = pd.DataFrame({
        "dataset_id": ["toy"] * 10,
        "replicate_id": [1] * 10,
        "classifier_id": ["c1"] * 5 + ["c2"] * 5,
        "fold_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "auc": [0.82, 0.80, 0.83, 0.79, 0.81, 0.80, 0.82, 0.81, 0.81, 0.81],
    })
    pw_zero = compute_power_c1_vs_c2(zero_diff_rep, truth_df, "toy")
    rule8_passed = (pw_zero["power"] == 0 and pw_zero["sign_error"] == 0 and "Rule 8" in pw_zero["outcome"])
    print(
        f"  - Mean Diff = 0.0 -> Outcome = '{pw_zero['outcome']}', Power = {pw_zero['power']} -> {'PASS' if rule8_passed else 'FAIL'}"
    )
    test_results.append(("12. Rule 8: Mean Paired Difference = 0", rule8_passed))

    # ==========================================================
    # TEST 13: Sample Size Independence Verification
    # ==========================================================
    print("\n[TEST 13] Testing Sample Size Independence (results.csv)...")
    sample_eval = evaluate_by_sample_size("truth.csv", "results.csv", [100, 500], verbose=False)
    has_100 = (100 in sample_eval) and (sample_eval[100]["valid_replicates_count"] == 20)
    has_500 = (500 in sample_eval) and (sample_eval[500]["valid_replicates_count"] == 20)
    rule13_passed = has_100 and has_500
    print(
        f"  - Independent evaluations: n_100 valid = {sample_eval[100]['valid_replicates_count']}, "
        f"n_500 valid = {sample_eval[500]['valid_replicates_count']} -> {'PASS' if rule13_passed else 'FAIL'}"
    )
    test_results.append(("13. Sample Size Independent Evaluation", rule13_passed))

    # ==========================================================
    # SUMMARY OF ALL TESTS
    # ==========================================================
    print("\n" + "=" * 78)
    print("COMPLETE TEST SUITE SUMMARY")
    print("=" * 78)
    all_passed = True
    for name, status in test_results:
        status_str = "PASS" if status else "FAIL"
        print(f"  {name:<55}: [{status_str}]")
        if not status:
            all_passed = False
    print("=" * 78)

    if all_passed:
        print("[ALL 13 TEST SUITES PASSED PERFECTLY!]")
    else:
        print("[SOME TESTS FAILED!]")
        sys.exit(1)


if __name__ == "__main__":
    # 1. Print detailed report on the 15-row toy replicate
    print("\n" + "#" * 78)
    print("STAGE 1: TOY DATASET VERIFICATION (toy_results.csv)")
    print("#" * 78)
    evaluate_dataset("truth.csv", "toy_results.csv", verbose=True)

    # 2. Run comprehensive automated test suite
    run_unit_tests()

    # 3. Evaluate the actual 600-row results.csv separately by sample size
    print("\n" + "#" * 78)
    print("STAGE 2: 600-ROW EXPERIMENTAL RESULTS (results.csv BY SAMPLE SIZE)")
    print("#" * 78)
    evaluate_by_sample_size("truth.csv", "results.csv", sample_sizes=[100, 500], verbose=True)
