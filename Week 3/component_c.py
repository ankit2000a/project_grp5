"""
CSIT332 Week 3 — Group 5 — Component C: The Statistical Layer
Core statistical engine implementing dual truth evaluation, multi-protocol aggregation,
lexicographical tie-breaking, Kendall's tau-b, optimism, and paired statistical tests
(Ordinary paired t-test and Nadeau-Bengio corrected test).
"""

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

CLASSIFIER_IDS = ["dtree", "gb", "knn", "logreg", "rf", "svm"]
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000]
PROTOCOLS = ["split80", "cv5", "repcv5x10"]


def resolve_filepath(relative_path: str, base_dir: Optional[str] = None) -> str:
    """Resolve file path relative to current module or explicit base directory."""
    if os.path.isabs(relative_path):
        return relative_path
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, relative_path))


def load_datasets(
    results_path: str = "data/reference_results_v2.csv",
    truth_path: str = "data/reference_truth_v2.csv",
    stability_path: Optional[str] = "data/reference_stability_v2.csv",
    base_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load results, ground truth, and optional stability reference datasets.
    Validates required column schemas and data integrity.
    """
    res_p = resolve_filepath(results_path, base_dir)
    tru_p = resolve_filepath(truth_path, base_dir)

    if not os.path.exists(res_p):
        raise FileNotFoundError(f"Results file not found: {res_p}")
    if not os.path.exists(tru_p):
        raise FileNotFoundError(f"Truth file not found: {tru_p}")

    results_df = pd.read_csv(res_p)
    truth_df = pd.read_csv(tru_p)

    # Validate results schema
    req_res_cols = [
        "dataset_id", "protocol_id", "n_sub", "replicate_id",
        "classifier_id", "repeat_id", "fold_id", "auc"
    ]
    for c in req_res_cols:
        if c not in results_df.columns:
            raise ValueError(f"Missing required column '{c}' in results dataset.")

    # Validate truth schema
    req_tru_cols = ["dataset_id", "classifier_id", "n_sub", "true_auc"]
    for c in req_tru_cols:
        if c not in truth_df.columns:
            raise ValueError(f"Missing required column '{c}' in truth dataset.")

    stability_df = None
    if stability_path:
        stab_p = resolve_filepath(stability_path, base_dir)
        if os.path.exists(stab_p):
            stability_df = pd.read_csv(stab_p)

    return results_df, truth_df, stability_df


def get_truth_subset(
    truth_df: pd.DataFrame,
    dataset_id: str = "synthetic_pilot",
    n_sub: Optional[Union[int, str]] = None,
    mode: str = "per_sample",
) -> pd.DataFrame:
    """
    Extract slice of ground truth for a given dataset and sample size.
    
    Parameters:
        truth_df: Full ground truth DataFrame.
        dataset_id: Target dataset ID.
        n_sub: Sample size (e.g. 50, 100, 200, 500, 1000, 2000).
        mode: 'per_sample' (Truth-at-n, Version A) or
              'asymptotic' (Asymptotic Truth, Version B).
    """
    sub = truth_df[truth_df["dataset_id"] == dataset_id].copy()
    sub["n_sub_str"] = sub["n_sub"].astype(str).str.strip().str.lower()

    if mode == "asymptotic":
        sub = sub[sub["n_sub_str"] == "asymptotic"]
    else:
        if n_sub is not None:
            sub = sub[sub["n_sub_str"] == str(n_sub).strip().lower()]

    return sub.drop(columns=["n_sub_str"])


def validate_replicate(
    replicate_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    protocol_id: str,
    n_sub: int,
    required_classifiers: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate whether a replicate satisfies all integrity and fold-completeness rules:
    - Protocol fold counts: split80 (1 row/clf), cv5 (5 rows/clf), repcv5x10 (50 rows/clf).
    - No NaN/infinite/non-computable AUCs or fold IDs.
    - All required classifiers present and matching truth.csv.
    """
    if replicate_df.empty:
        return False, "Replicate dataframe is empty."

    expected_rows_per_clf = {
        "split80": 1,
        "cv5": 5,
        "repcv5x10": 50,
    }.get(protocol_id)

    if expected_rows_per_clf is None:
        return False, f"Unknown protocol '{protocol_id}'."

    truth_subset = get_truth_subset(truth_df, dataset_id=replicate_df["dataset_id"].iloc[0], n_sub=n_sub, mode="per_sample")
    if truth_subset.empty:
        return False, f"No ground truth entry for dataset at n={n_sub}."

    if required_classifiers is None:
        required_classifiers = sorted(truth_subset["classifier_id"].unique().tolist())

    replicate_clfs = set(replicate_df["classifier_id"].dropna().unique())
    missing = set(required_classifiers) - replicate_clfs
    if missing:
        return False, f"Missing required classifiers: {sorted(list(missing))}."

    for clf in required_classifiers:
        clf_rows = replicate_df[replicate_df["classifier_id"] == clf]
        if len(clf_rows) != expected_rows_per_clf:
            return False, f"Classifier '{clf}' has {len(clf_rows)} rows, expected {expected_rows_per_clf}."

        if clf_rows["auc"].isna().any() or np.isinf(clf_rows["auc"]).any():
            return False, f"Classifier '{clf}' contains NaN or infinite fold AUC."

        if protocol_id in ["cv5", "repcv5x10"]:
            fold_ids = clf_rows["fold_id"].to_numpy(dtype=float)
            if not np.all(np.equal(np.mod(fold_ids, 1), 0)):
                return False, f"Classifier '{clf}' contains non-integer fold IDs."

    return True, "Valid replicate."


def calculate_replicate_means(replicate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each classifier's replicate-level observed AUC:
    Mean of all its rows under that protocol:
    - split80: 1 row -> mean = auc
    - cv5: 5 rows -> mean across 5 folds
    - repcv5x10: 50 rows -> mean across 50 folds (10 repeats x 5 folds)
    """
    summary = (
        replicate_df.groupby(["dataset_id", "protocol_id", "n_sub", "replicate_id", "classifier_id"])
        .agg(
            mean_auc=("auc", "mean"),
            fold_count=("auc", "count"),
        )
        .reset_index()
    )
    return summary


def get_declared_winner(replicate_summary: pd.DataFrame) -> Tuple[str, float]:
    """
    Determine the declared winner based on highest observed mean AUC.
    Lexicographical tie-break: Exact ties in mean AUC are broken by choosing the
    lexicographically smallest classifier_id (ascending alphabetical order).
    """
    sorted_df = replicate_summary.sort_values(
        by=["mean_auc", "classifier_id"], ascending=[False, True]
    )
    winner_row = sorted_df.iloc[0]
    return str(winner_row["classifier_id"]), float(winner_row["mean_auc"])


def get_true_best_classifiers(truth_subset: pd.DataFrame) -> Tuple[List[str], float, bool]:
    """
    Determine genuinely best classifier(s) from a truth subset.
    Truth-Tie Rule: Returns all classifiers sharing the maximum true AUC (within 1e-9).
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
    Declared winner evaluated against the true best classifier(s).
    Truth-Tie Rule: If declared winner is ANY of the tied true best classifiers, correct = 1.
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
    Compute Kendall's tau-b rank correlation between observed mean AUC and true AUC.
    Explicitly accounts for ties in observed scores and true scores.
    """
    merged = pd.merge(replicate_summary, truth_subset, on="classifier_id")
    merged = merged.sort_values(by="classifier_id").reset_index(drop=True)
    if len(merged) < 2:
        return {"tau": 0.0, "p_value": 1.0}

    tau_res = stats.kendalltau(merged["true_auc"], merged["mean_auc"], variant="b")
    tau_val = float(tau_res.statistic) if not np.isnan(tau_res.statistic) else 0.0
    p_val = float(tau_res.pvalue) if not np.isnan(tau_res.pvalue) else 1.0
    return {"tau": tau_val, "p_value": p_val}


def compute_optimism(
    replicate_summary: pd.DataFrame,
    truth_subset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Selection Optimism:
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


def compute_paired_tests(
    replicate_df: pd.DataFrame,
    truth_subset: pd.DataFrame,
    c1_id: str,
    c2_id: str,
    protocol_id: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform both Ordinary paired t-test and Nadeau-Bengio corrected test for (c1 vs c2).
    
    Formulas:
    - Ordinary:
        t_ord = mean(d) / sqrt(var(d) / J), df = J - 1
    - Nadeau-Bengio corrected:
        t_nb = mean(d) / sqrt((1/J + 1/4) * var(d)), df = J - 1
        where J = 5 (cv5) or 50 (repcv5x10), and 1/4 = n_test / n_train (5-fold CV).
    
    Evaluates:
    - true_gap = true_auc(c1) - true_auc(c2)
    - Power: p < alpha and sign(mean(d)) == sign(true_gap)
    - Sign-Error: p < alpha and sign(mean(d)) == -sign(true_gap)
    - Type I Error (if true_gap == 0): p < alpha
    """
    if protocol_id not in ["cv5", "repcv5x10"]:
        raise ValueError(f"Significance testing not applicable for protocol '{protocol_id}'.")

    J = 5 if protocol_id == "cv5" else 50

    c1_folds = replicate_df[replicate_df["classifier_id"] == c1_id].sort_values(
        by=["repeat_id", "fold_id"]
    )
    c2_folds = replicate_df[replicate_df["classifier_id"] == c2_id].sort_values(
        by=["repeat_id", "fold_id"]
    )

    c1_truth = truth_subset[truth_subset["classifier_id"] == c1_id]
    c2_truth = truth_subset[truth_subset["classifier_id"] == c2_id]
    c1_true = float(c1_truth["true_auc"].iloc[0]) if not c1_truth.empty else np.nan
    c2_true = float(c2_truth["true_auc"].iloc[0]) if not c2_truth.empty else np.nan

    true_gap = c1_true - c2_true
    is_truth_tie = np.isclose(c1_true, c2_true, atol=1e-9)

    base_res = {
        "c1": c1_id,
        "c2": c2_id,
        "true_gap": true_gap,
        "abs_gap": abs(true_gap),
        "mean_diff": 0.0,
        "var_diff": 0.0,
        "df": J - 1,
        # Ordinary test outputs
        "ord_t": np.nan,
        "ord_p": 1.0,
        "ord_power": 0,
        "ord_sign_error": 0,
        "ord_type1_error": 0,
        # Nadeau-Bengio test outputs
        "nb_t": np.nan,
        "nb_p": 1.0,
        "nb_power": 0,
        "nb_sign_error": 0,
        "nb_type1_error": 0,
    }

    if len(c1_folds) != J or len(c2_folds) != J:
        return base_res

    diffs = c1_folds["auc"].to_numpy(dtype=float) - c2_folds["auc"].to_numpy(dtype=float)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))

    base_res["mean_diff"] = mean_d
    base_res["var_diff"] = var_d

    # Degenerate cases: zero variance or mean diff == 0
    if np.isclose(var_d, 0.0, atol=1e-12) or np.isclose(mean_d, 0.0, atol=1e-12):
        return base_res

    # 1. Ordinary paired t-test
    se_ord = np.sqrt(var_d / J)
    t_ord = mean_d / se_ord
    p_ord = 2.0 * stats.t.sf(abs(t_ord), df=J - 1)
    base_res["ord_t"] = float(t_ord)
    base_res["ord_p"] = float(p_ord)

    # 2. Nadeau-Bengio corrected test
    se_nb = np.sqrt((1.0 / J + 0.25) * var_d)
    t_nb = mean_d / se_nb
    p_nb = 2.0 * stats.t.sf(abs(t_nb), df=J - 1)
    base_res["nb_t"] = float(t_nb)
    base_res["nb_p"] = float(p_nb)

    # Evaluate detection for Ordinary
    if p_ord < alpha:
        if is_truth_tie:
            base_res["ord_type1_error"] = 1
        elif (true_gap > 0 and mean_d > 0) or (true_gap < 0 and mean_d < 0):
            base_res["ord_power"] = 1
        elif (true_gap > 0 and mean_d < 0) or (true_gap < 0 and mean_d > 0):
            base_res["ord_sign_error"] = 1

    # Evaluate detection for Nadeau-Bengio
    if p_nb < alpha:
        if is_truth_tie:
            base_res["nb_type1_error"] = 1
        elif (true_gap > 0 and mean_d > 0) or (true_gap < 0 and mean_d < 0):
            base_res["nb_power"] = 1
        elif (true_gap > 0 and mean_d < 0) or (true_gap < 0 and mean_d > 0):
            base_res["nb_sign_error"] = 1

    return base_res


def evaluate_protocol_summary(
    results_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    protocol_id: str,
    truth_mode: str = "per_sample",
    replicate_range: Optional[Tuple[int, int]] = None,
    sample_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute the four statistical quantities across all applicable sample sizes for a given protocol.
    
    Parameters:
        results_df: Full reference results DataFrame.
        truth_df: Full ground truth DataFrame.
        protocol_id: 'split80', 'cv5', or 'repcv5x10'.
        truth_mode: 'per_sample' (truth-at-n) or 'asymptotic' (asymptotic truth).
        replicate_range: Optional (min_rep, max_rep) to filter replicates (e.g. (1, 50) for paired comparison).
        sample_sizes: Optional list of sample sizes to evaluate.
    """
    proto_df = results_df[results_df["protocol_id"] == protocol_id]
    if sample_sizes is None:
        sample_sizes = sorted(proto_df["n_sub"].unique().tolist())

    records = []
    for n in sample_sizes:
        sub_n = proto_df[proto_df["n_sub"] == n]
        if sub_n.empty:
            continue

        truth_sub = get_truth_subset(truth_df, dataset_id=sub_n["dataset_id"].iloc[0], n_sub=n, mode=truth_mode)
        if truth_sub.empty:
            continue

        # Group by replicate
        rep_ids = sorted(sub_n["replicate_id"].unique().tolist())
        if replicate_range is not None:
            min_r, max_r = replicate_range
            rep_ids = [r for r in rep_ids if min_r <= r <= max_r]

        valid_reps_count = 0
        dropped_reps_count = 0
        wc_list = []
        tau_list = []
        opt_list = []

        for r in rep_ids:
            rep_df = sub_n[sub_n["replicate_id"] == r]
            is_valid, _ = validate_replicate(rep_df, truth_df, protocol_id=protocol_id, n_sub=n)
            if not is_valid:
                dropped_reps_count += 1
                continue

            valid_reps_count += 1
            rep_summary = calculate_replicate_means(rep_df)

            wc_res = compute_winner_correct(rep_summary, truth_sub)
            tau_res = compute_kendall_tau(rep_summary, truth_sub)
            opt_res = compute_optimism(rep_summary, truth_sub)

            wc_list.append(wc_res["is_correct"])
            tau_list.append(tau_res["tau"])
            opt_list.append(opt_res["optimism"])

        records.append({
            "protocol_id": protocol_id,
            "truth_mode": truth_mode,
            "n_sub": n,
            "total_replicates": len(rep_ids),
            "valid_replicates": valid_reps_count,
            "dropped_replicates": dropped_reps_count,
            "winner_correct": float(np.mean(wc_list)) if wc_list else np.nan,
            "kendall_tau": float(np.mean(tau_list)) if tau_list else np.nan,
            "optimism": float(np.mean(opt_list)) if opt_list else np.nan,
        })

    return pd.DataFrame(records)
