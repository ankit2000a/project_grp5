"""
CSIT332 Week 3 — Group 5 — Component C: Automated Verification Test Suite
Thoroughly validates all statistical formulas, tie-breaking rules, multi-protocol aggregations,
Nadeau-Bengio corrections, candidate subset counts, and boundary conditions.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

# Ensure Week 3 directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from component_c import (
    CLASSIFIER_IDS,
    SAMPLE_SIZES,
    load_datasets,
    get_truth_subset,
    validate_replicate,
    calculate_replicate_means,
    get_declared_winner,
    get_true_best_classifiers,
    compute_winner_correct,
    compute_kendall_tau,
    compute_optimism,
    compute_paired_tests,
    evaluate_protocol_summary,
)
from candidate_analysis import run_candidate_count_analysis
from power_calculator import run_all_pair_tests, build_calculator_table


def run_all_verification_tests():
    print("=" * 80)
    print("CSIT332 WEEK 3 — COMPONENT C: STATISTICAL LAYER AUTOMATED TEST SUITE")
    print("=" * 80)

    passed_count = 0
    total_count = 16

    def report(test_num: int, name: str, success: bool, detail: str = ""):
        nonlocal passed_count
        status = "PASS" if success else "FAIL"
        print(f"[TEST {test_num:02d}] [{status}] {name}")
        if detail:
            print(f"           Detail: {detail}")
        if success:
            passed_count += 1
        else:
            raise AssertionError(f"Test {test_num} FAILED: {detail}")

    # -------------------------------------------------------------------------
    # TEST 01: Dual Truth Loading & Version Filtering
    # -------------------------------------------------------------------------
    results_df, truth_df, _ = load_datasets()
    t_50 = get_truth_subset(truth_df, n_sub=50, mode="per_sample")
    t_asymp = get_truth_subset(truth_df, mode="asymptotic")

    t1_pass = (
        len(t_50) == 6
        and len(t_asymp) == 6
        and set(t_50["classifier_id"]) == set(CLASSIFIER_IDS)
        and set(t_asymp["classifier_id"]) == set(CLASSIFIER_IDS)
        and float(t_50[t_50["classifier_id"] == "dtree"]["true_auc"].iloc[0]) < float(t_asymp[t_asymp["classifier_id"] == "dtree"]["true_auc"].iloc[0])
    )
    report(1, "Dual Truth Loading & Version Filtering", t1_pass,
           "Verified truth-at-50 (finite) vs asymptotic truth slices for all 6 classifiers.")

    # -------------------------------------------------------------------------
    # TEST 02: Protocol Replicate Fold Aggregation
    # -------------------------------------------------------------------------
    # split80 (1 row), cv5 (5 rows), repcv5x10 (50 rows)
    rep_split = results_df[(results_df["protocol_id"] == "split80") & (results_df["n_sub"] == 100) & (results_df["replicate_id"] == 1)]
    rep_cv5 = results_df[(results_df["protocol_id"] == "cv5") & (results_df["n_sub"] == 100) & (results_df["replicate_id"] == 1)]
    rep_repcv = results_df[(results_df["protocol_id"] == "repcv5x10") & (results_df["n_sub"] == 100) & (results_df["replicate_id"] == 1)]

    means_split = calculate_replicate_means(rep_split)
    means_cv5 = calculate_replicate_means(rep_cv5)
    means_repcv = calculate_replicate_means(rep_repcv)

    t2_pass = (
        (means_split["fold_count"] == 1).all()
        and (means_cv5["fold_count"] == 5).all()
        and (means_repcv["fold_count"] == 50).all()
        and len(means_split) == 6
        and len(means_cv5) == 6
        and len(means_repcv) == 6
    )
    report(2, "Protocol-Specific Fold Count Aggregation", t2_pass,
           "Verified replicate means correctly aggregate 1 row (split80), 5 rows (cv5), and 50 rows (repcv5x10).")

    # -------------------------------------------------------------------------
    # TEST 03: Lexicographical Tie-Breaking on Mean AUC
    # -------------------------------------------------------------------------
    tied_df = pd.DataFrame([
        {"dataset_id": "test", "protocol_id": "cv5", "n_sub": 100, "replicate_id": 1, "classifier_id": "svm", "mean_auc": 0.820},
        {"dataset_id": "test", "protocol_id": "cv5", "n_sub": 100, "replicate_id": 1, "classifier_id": "logreg", "mean_auc": 0.820},
        {"dataset_id": "test", "protocol_id": "cv5", "n_sub": 100, "replicate_id": 1, "classifier_id": "dtree", "mean_auc": 0.700},
    ])
    winner, winner_auc = get_declared_winner(tied_df)
    t3_pass = (winner == "logreg" and winner_auc == 0.820)
    report(3, "Lexicographical Tie-Breaking on Empirical Mean AUC", t3_pass,
           f"Tied svm & logreg at 0.820 correctly broken by alphabetical priority -> '{winner}'.")

    # -------------------------------------------------------------------------
    # TEST 04: Truth-Tie Rule in Winner-Correct Rate
    # -------------------------------------------------------------------------
    # Truth has two classifiers with equal true AUC
    truth_tie_df = pd.DataFrame([
        {"dataset_id": "test", "classifier_id": "gb", "n_sub": "100", "true_auc": 0.850},
        {"dataset_id": "test", "classifier_id": "rf", "n_sub": "100", "true_auc": 0.850},
        {"dataset_id": "test", "classifier_id": "dtree", "n_sub": "100", "true_auc": 0.700},
    ])
    best_clfs, max_auc, is_tie = get_true_best_classifiers(truth_tie_df)
    res_gb_wins = compute_winner_correct(
        pd.DataFrame([{"dataset_id": "test", "classifier_id": "gb", "mean_auc": 0.88}]),
        truth_tie_df
    )
    res_rf_wins = compute_winner_correct(
        pd.DataFrame([{"dataset_id": "test", "classifier_id": "rf", "mean_auc": 0.88}]),
        truth_tie_df
    )
    res_dtree_wins = compute_winner_correct(
        pd.DataFrame([{"dataset_id": "test", "classifier_id": "dtree", "mean_auc": 0.88}]),
        truth_tie_df
    )
    t4_pass = (
        is_tie is True
        and set(best_clfs) == {"gb", "rf"}
        and res_gb_wins["is_correct"] == 1
        and res_rf_wins["is_correct"] == 1
        and res_dtree_wins["is_correct"] == 0
    )
    report(4, "Truth-Tie Handling in Winner-Correct Evaluation", t4_pass,
           "Declared winner matching ANY tied true best counts as correct = 1.")

    # -------------------------------------------------------------------------
    # TEST 05: Invalid Replicate & Non-Computable Fold Filtering
    # -------------------------------------------------------------------------
    corrupt_rep = rep_cv5.copy()
    corrupt_rep.loc[corrupt_rep["classifier_id"] == "svm", "auc"] = np.nan
    is_valid_nan, msg_nan = validate_replicate(corrupt_rep, truth_df, protocol_id="cv5", n_sub=100)

    incomplete_rep = rep_cv5[rep_cv5["fold_id"] != 5].copy()
    is_valid_inc, msg_inc = validate_replicate(incomplete_rep, truth_df, protocol_id="cv5", n_sub=100)

    t5_pass = (is_valid_nan is False and is_valid_inc is False)
    report(5, "Invalid Replicate Rejection (NaN / Incomplete Folds)", t5_pass,
           "Correctly rejects NaN fold AUC and incomplete 4-fold replicate.")

    # -------------------------------------------------------------------------
    # TEST 06: Selection Optimism Calculation
    # -------------------------------------------------------------------------
    rep_summary_test = pd.DataFrame([
        {"dataset_id": "synthetic_pilot", "classifier_id": "rf", "mean_auc": 0.800},
        {"dataset_id": "synthetic_pilot", "classifier_id": "dtree", "mean_auc": 0.700},
    ])
    truth_sub_test = pd.DataFrame([
        {"dataset_id": "synthetic_pilot", "classifier_id": "rf", "true_auc": 0.750},
        {"dataset_id": "synthetic_pilot", "classifier_id": "dtree", "true_auc": 0.650},
    ])
    opt_res = compute_optimism(rep_summary_test, truth_sub_test)
    t6_pass = np.isclose(opt_res["optimism"], 0.050, atol=1e-9) and opt_res["declared_winner"] == "rf"
    report(6, "Selection Optimism Definition & Precision", t6_pass,
           f"Optimism = 0.800 - 0.750 = {opt_res['optimism']:.3f} exactly.")

    # -------------------------------------------------------------------------
    # TEST 07: Kendall's Tau-b Tie Adjustment
    # -------------------------------------------------------------------------
    kt_res = compute_kendall_tau(rep_summary_test, truth_sub_test)
    t7_pass = np.isclose(kt_res["tau"], 1.0, atol=1e-9)
    report(7, "Kendall's Tau-b Rank Concordance", t7_pass,
           f"Perfect ranking concordance yields tau = {kt_res['tau']:.1f}.")

    # -------------------------------------------------------------------------
    # TEST 08: Nadeau-Bengio Variance Inflation & Formula Verification
    # -------------------------------------------------------------------------
    # In 5-fold CV: J=5, se_ord = sqrt(var/5), se_nb = sqrt((1/5 + 1/4)*var) = sqrt(0.45*var)
    # se_nb / se_ord = sqrt(0.45 / 0.20) = sqrt(2.25) = 1.500 exactly!
    # Therefore, t_nb = t_ord / 1.500 exactly!
    test_rep = rep_cv5.copy()
    nb_check = compute_paired_tests(test_rep, t_50, "gb", "knn", protocol_id="cv5")
    se_ratio = nb_check["ord_t"] / nb_check["nb_t"]
    t8_pass = np.isclose(se_ratio, 1.500, atol=1e-4) and (nb_check["nb_p"] > nb_check["ord_p"])
    report(8, "Nadeau-Bengio Variance Inflation Ratio (Exact 1.500x in CV5)", t8_pass,
           f"Confirmed t_ord / t_nb = {se_ratio:.4f} (theoretical: sqrt(0.45 / 0.20) = 1.500).")

    # -------------------------------------------------------------------------
    # TEST 09: Degenerate Paired Tests (Zero Variance & Zero Mean Diff)
    # -------------------------------------------------------------------------
    degen_rep = rep_cv5.copy()
    # Set identical AUCs across folds
    degen_rep.loc[degen_rep["classifier_id"] == "gb", "auc"] = 0.75
    degen_rep.loc[degen_rep["classifier_id"] == "knn", "auc"] = 0.75
    degen_res = compute_paired_tests(degen_rep, t_50, "gb", "knn", protocol_id="cv5")
    t9_pass = (degen_res["ord_power"] == 0 and degen_res["nb_power"] == 0 and degen_res["mean_diff"] == 0.0)
    report(9, "Degenerate Paired Tests Safe Handling", t9_pass,
           "Zero fold variance and zero mean diff cleanly handled without divide-by-zero.")

    # -------------------------------------------------------------------------
    # TEST 10: Candidate-Count Subsets Math (k = 2, 3, 4, 5, 6)
    # -------------------------------------------------------------------------
    import itertools
    subsets_k2 = list(itertools.combinations(CLASSIFIER_IDS, 2))
    subsets_k3 = list(itertools.combinations(CLASSIFIER_IDS, 3))
    subsets_k4 = list(itertools.combinations(CLASSIFIER_IDS, 4))
    subsets_k5 = list(itertools.combinations(CLASSIFIER_IDS, 5))
    subsets_k6 = list(itertools.combinations(CLASSIFIER_IDS, 6))

    t10_pass = (
        len(subsets_k2) == 15
        and len(subsets_k3) == 20
        and len(subsets_k4) == 15
        and len(subsets_k5) == 6
        and len(subsets_k6) == 1
    )
    report(10, "Candidate-Count Combinatorial Subsets Enumeration", t10_pass,
           "Subsets: k=2 (15), k=3 (20), k=4 (15), k=5 (6), k=6 (1) -> Total: 57 subsets.")

    # -------------------------------------------------------------------------
    # TEST 11: Task 2 Candidate Analysis Execution & Optimism Growth
    # -------------------------------------------------------------------------
    cand_summary, cand_details = run_candidate_count_analysis(results_df, truth_df)
    # Verify optimism grows with k at n=50
    opt_n50 = cand_summary[cand_summary["n_sub"] == 50].sort_values(by="k")["optimism_mean"].tolist()
    is_monotonic_opt = all(x < y for x, y in zip(opt_n50, opt_n50[1:]))

    t11_pass = (len(cand_summary) == 30 and is_monotonic_opt) # 6 sizes x 5 k-values = 30 rows
    report(11, "Candidate-Count Analysis & Monotonic Optimism Growth", t11_pass,
           f"Optimism at n=50 grows monotonically: {[round(x, 4) for x in opt_n50]}.")

    # -------------------------------------------------------------------------
    # TEST 12: Task 3 Total Conditions Count (Exactly 300 Conditions)
    # -------------------------------------------------------------------------
    pair_tests_df = run_all_pair_tests(results_df, truth_df)
    cv5_conds = len(pair_tests_df[pair_tests_df["protocol_id"] == "cv5"])
    repcv_conds = len(pair_tests_df[pair_tests_df["protocol_id"] == "repcv5x10"])

    t12_pass = (
        len(pair_tests_df) == 300
        and cv5_conds == 180   # 15 pairs x 6 sizes x 2 tests
        and repcv_conds == 120 # 15 pairs x 4 sizes x 2 tests
    )
    report(12, "Pair-Level Power Analysis Exact Condition Count (300 Total)", t12_pass,
           f"CV5: {cv5_conds} conditions, repcv5x10: {repcv_conds} conditions -> Exactly 300.")

    # -------------------------------------------------------------------------
    # TEST 13: Task 4 Calculator Table & 'Never reached by n=2000' Detection
    # -------------------------------------------------------------------------
    calc_table = build_calculator_table(pair_tests_df)
    # Check gap=0.01 under nadeau_bengio is 'Never reached by n=2000'
    gap01_nb = calc_table[(calc_table["protocol"] == "cv5") & (calc_table["test"] == "nadeau_bengio") & (calc_table["true_gap"] == 0.01)]
    status_str = gap01_nb["smallest_n_for_80pct_power"].iloc[0]

    t13_pass = ("Never reached by n=2000" in status_str and len(calc_table) == 12)
    report(13, "First Calculator Table & Non-Convergence Labeling", t13_pass,
           f"Gap=0.01 under Nadeau-Bengio correctly labels: '{status_str}'.")

    # -------------------------------------------------------------------------
    # TEST 14: Protocol Summary Table Execution on Reference Data
    # -------------------------------------------------------------------------
    proto_split = evaluate_protocol_summary(results_df, truth_df, protocol_id="split80")
    proto_cv5 = evaluate_protocol_summary(results_df, truth_df, protocol_id="cv5")
    proto_repcv = evaluate_protocol_summary(results_df, truth_df, protocol_id="repcv5x10")

    t14_pass = (
        len(proto_split) == 6
        and len(proto_cv5) == 6
        and len(proto_repcv) == 4
        and (proto_cv5["winner_correct"] > 0).all()
        and (proto_cv5["kendall_tau"] > 0).all()
    )
    report(14, "Multi-Protocol Full Reference Dataset Execution", t14_pass,
           "Successfully evaluated all protocols across all sample sizes without missing data.")

    # -------------------------------------------------------------------------
    # TEST 15: repcv5x10 Nadeau-Bengio SE Inflation Ratio (sqrt(13.5) ≈ 3.6742)
    # -------------------------------------------------------------------------
    # In repcv5x10: J=50, se_ord = sqrt(var/50) = sqrt(0.02*var)
    # se_nb = sqrt((1/50 + 1/4)*var) = sqrt((0.02 + 0.25)*var) = sqrt(0.27*var)
    # se_nb / se_ord = sqrt(0.27 / 0.02) = sqrt(13.5) ≈ 3.6742346...
    # Therefore, t_ord / t_nb = sqrt(13.5) exactly!
    test_repcv = rep_repcv.copy()
    nb_check_repcv = compute_paired_tests(test_repcv, t_50, "gb", "knn", protocol_id="repcv5x10")
    se_ratio_repcv = nb_check_repcv["ord_t"] / nb_check_repcv["nb_t"]
    expected_ratio_repcv = np.sqrt(13.5)
    t15_pass = np.isclose(se_ratio_repcv, expected_ratio_repcv, atol=1e-4) and (nb_check_repcv["nb_p"] > nb_check_repcv["ord_p"])
    report(15, "repcv5x10 Nadeau-Bengio SE Inflation Ratio (sqrt(13.5) ≈ 3.6742)", t15_pass,
           f"Confirmed t_ord / t_nb = {se_ratio_repcv:.4f} (theoretical: sqrt(0.27 / 0.02) = {expected_ratio_repcv:.4f}).")

    # -------------------------------------------------------------------------
    # TEST 16: Explicit split80 Rejection Guard in Paired Tests
    # -------------------------------------------------------------------------
    split80_rejected = False
    try:
        compute_paired_tests(rep_split, t_50, "gb", "knn", protocol_id="split80")
    except ValueError as e:
        if "significance testing not applicable" in str(e).lower() or "split80" in str(e).lower():
            split80_rejected = True
    t16_pass = split80_rejected
    report(16, "Explicit split80 Exclusion Guard in Paired Significance Tests", t16_pass,
           "Confirmed compute_paired_tests raises ValueError when called with protocol_id='split80'.")

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {passed_count}/{total_count} TEST SUITES PASSED (100%)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_verification_tests()
