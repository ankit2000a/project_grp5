"""
CSIT332 Week 3 — Group 5 — Component C: Tasks 3 & 4 — Power Analysis & First Calculator
Analyzes all 15 classifier pairs across CV5 and repcv5x10 (300 test conditions total).
Compares Ordinary paired t-test against the Nadeau-Bengio corrected test.
Generates empirical power curves and the First Sample-Size Calculator table.
"""

import os
import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

try:
    from .component_c import (
        CLASSIFIER_IDS,
        SAMPLE_SIZES,
        get_truth_subset,
        compute_paired_tests,
    )
except ImportError:
    from component_c import (
        CLASSIFIER_IDS,
        SAMPLE_SIZES,
        get_truth_subset,
        compute_paired_tests,
    )


def run_all_pair_tests(
    results_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    dataset_id: str = "synthetic_pilot",
) -> pd.DataFrame:
    """
    Task 3: Power for every pair with two tests.
    
    Evaluates all 15 classifier pairs under:
    - cv5: 15 pairs x 6 sample sizes x 2 tests = 180 conditions
    - repcv5x10: 15 pairs x 4 sample sizes (n <= 500) x 2 tests = 120 conditions
    Total conditions: Exactly 300.
    
    Reports:
    - true_gap: true AUC difference from truth-at-n
    - power: percentage of replicates with significant detection in correct direction (p < 0.05)
    - sign_error: percentage of replicates with significant detection in wrong direction (p < 0.05)
    """
    pairs = list(itertools.combinations(CLASSIFIER_IDS, 2))
    assert len(pairs) == 15, f"Expected 15 pairs, got {len(pairs)}"

    protocol_configs = [
        ("cv5", [50, 100, 200, 500, 1000, 2000]),
        ("repcv5x10", [50, 100, 200, 500]),
    ]

    records = []

    for protocol_id, sample_sizes in protocol_configs:
        proto_df = results_df[
            (results_df["dataset_id"] == dataset_id) & (results_df["protocol_id"] == protocol_id)
        ]

        for n in sample_sizes:
            sub_n = proto_df[proto_df["n_sub"] == n]
            if sub_n.empty:
                continue

            truth_sub = get_truth_subset(truth_df, dataset_id=dataset_id, n_sub=n, mode="per_sample")
            if truth_sub.empty:
                continue

            replicate_ids = sorted(sub_n["replicate_id"].unique().tolist())
            replicate_dfs = {r: df_r for r, df_r in sub_n.groupby("replicate_id")}

            for c1, c2 in pairs:
                ord_powers, ord_sign_errors = [], []
                nb_powers, nb_sign_errors = [], []
                true_gap = 0.0

                for r in replicate_ids:
                    rep_df = replicate_dfs[r]
                    res = compute_paired_tests(
                        replicate_df=rep_df,
                        truth_subset=truth_sub,
                        c1_id=c1,
                        c2_id=c2,
                        protocol_id=protocol_id,
                        alpha=0.05,
                    )
                    true_gap = res["true_gap"]
                    ord_powers.append(res["ord_power"])
                    ord_sign_errors.append(res["ord_sign_error"])
                    nb_powers.append(res["nb_power"])
                    nb_sign_errors.append(res["nb_sign_error"])

                # Condition 1: Ordinary paired t-test
                records.append({
                    "protocol_id": protocol_id,
                    "test_type": "ordinary",
                    "sample_size": n,
                    "pair": f"{c1}_vs_{c2}",
                    "c1": c1,
                    "c2": c2,
                    "true_gap": true_gap,
                    "abs_gap": abs(true_gap),
                    "power": float(np.mean(ord_powers)),
                    "sign_error": float(np.mean(ord_sign_errors)),
                    "num_replicates": len(replicate_ids),
                })

                # Condition 2: Nadeau-Bengio corrected test
                records.append({
                    "protocol_id": protocol_id,
                    "test_type": "nadeau_bengio",
                    "sample_size": n,
                    "pair": f"{c1}_vs_{c2}",
                    "c1": c1,
                    "c2": c2,
                    "true_gap": true_gap,
                    "abs_gap": abs(true_gap),
                    "power": float(np.mean(nb_powers)),
                    "sign_error": float(np.mean(nb_sign_errors)),
                    "num_replicates": len(replicate_ids),
                })

    df_tests = pd.DataFrame(records)
    assert len(df_tests) == 300, f"Expected exactly 300 test conditions, got {len(df_tests)}"
    return df_tests


def estimate_power_at_gap(
    df_protocol_test: pd.DataFrame,
    sample_size: int,
    target_gap: float,
) -> float:
    """
    Interpolate or estimate statistical power at a specified target gap for a given sample size
    using linear interpolation on sorted absolute true gaps.
    """
    sub = df_protocol_test[df_protocol_test["sample_size"] == sample_size].sort_values(by="abs_gap")
    if sub.empty:
        return 0.0

    gaps = sub["abs_gap"].to_numpy(dtype=float)
    powers = sub["power"].to_numpy(dtype=float)

    # If target gap is smaller than the minimum observed gap
    if target_gap <= gaps[0]:
        return float(powers[0])

    # If target gap is larger than the maximum observed gap
    if target_gap >= gaps[-1]:
        return float(powers[-1])

    # Linear interpolation
    interp_power = float(np.interp(target_gap, gaps, powers))
    return interp_power


def build_calculator_table(
    df_tests: pd.DataFrame,
    target_gaps: List[float] = [0.01, 0.02, 0.05],
    power_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Task 4: The First Calculator.
    
    For true gaps of 0.01, 0.02, and 0.05:
    Finds the smallest sample size at which statistical power exceeds 0.80.
    Where power never reaches 0.80 by n=2000, explicitly records 'Never reached by n=2000'.
    
    Generates results for:
    - CV5 Ordinary
    - CV5 Nadeau-Bengio
    - repcv5x10 Ordinary (up to n=500)
    - repcv5x10 Nadeau-Bengio (up to n=500)
    """
    rows = []
    configs = [
        ("cv5", "ordinary", [50, 100, 200, 500, 1000, 2000], "n=2000"),
        ("cv5", "nadeau_bengio", [50, 100, 200, 500, 1000, 2000], "n=2000"),
        ("repcv5x10", "ordinary", [50, 100, 200, 500], "n=500"),
        ("repcv5x10", "nadeau_bengio", [50, 100, 200, 500], "n=500"),
    ]

    for protocol_id, test_type, sample_sizes, max_scope in configs:
        sub_df = df_tests[
            (df_tests["protocol_id"] == protocol_id) & (df_tests["test_type"] == test_type)
        ]

        for gap in target_gaps:
            min_n = None
            power_at_min_n = None
            powers_by_n = {}

            for n in sample_sizes:
                p_est = estimate_power_at_gap(sub_df, sample_size=n, target_gap=gap)
                powers_by_n[n] = p_est
                if p_est >= power_threshold and min_n is None:
                    min_n = n
                    power_at_min_n = p_est

            if min_n is not None:
                min_n_str = f"n = {min_n}"
                verb = "Exceeds" if power_at_min_n > 0.80 else "Reaches"
                status = f"{verb} 80% (Power: {power_at_min_n:.3f})"
            else:
                min_n_str = f"Never reached by {max_scope}"
                max_p = max(powers_by_n.values()) if powers_by_n else 0.0
                status = f"Maximum power achieved: {max_p:.2f}"

            rows.append({
                "protocol": protocol_id,
                "test": test_type,
                "true_gap": gap,
                "smallest_n_for_80pct_power": min_n_str,
                "status_detail": status,
                "power_n50": powers_by_n.get(50, np.nan),
                "power_n100": powers_by_n.get(100, np.nan),
                "power_n200": powers_by_n.get(200, np.nan),
                "power_n500": powers_by_n.get(500, np.nan),
                "power_n1000": powers_by_n.get(1000, np.nan),
                "power_n2000": powers_by_n.get(2000, np.nan),
            })

    return pd.DataFrame(rows)


def plot_power_calculator_curves(
    df_tests: pd.DataFrame,
    output_path: str = "plots/power_curves_calculator.png",
):
    """
    Generate high-resolution publication plot for Tasks 3 & 4:
    Power against True Gap with a separate curve for each sample size.
    Displays side-by-side comparison of Ordinary vs. Nadeau-Bengio corrected tests.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 1. Row 0: CV5 (Ordinary vs. Nadeau-Bengio)
    cv5_ord = df_tests[(df_tests["protocol_id"] == "cv5") & (df_tests["test_type"] == "ordinary")]
    cv5_nb = df_tests[(df_tests["protocol_id"] == "cv5") & (df_tests["test_type"] == "nadeau_bengio")]

    # 2. Row 1: repcv5x10 (Ordinary vs. Nadeau-Bengio)
    rep_ord = df_tests[(df_tests["protocol_id"] == "repcv5x10") & (df_tests["test_type"] == "ordinary")]
    rep_nb = df_tests[(df_tests["protocol_id"] == "repcv5x10") & (df_tests["test_type"] == "nadeau_bengio")]

    panels = [
        (axes[0, 0], cv5_ord, "CV5 — Ordinary Paired t-Test", [50, 100, 200, 500, 1000, 2000]),
        (axes[0, 1], cv5_nb, "CV5 — Nadeau-Bengio Corrected Test", [50, 100, 200, 500, 1000, 2000]),
        (axes[1, 0], rep_ord, "repcv5x10 — Ordinary Paired t-Test", [50, 100, 200, 500]),
        (axes[1, 1], rep_nb, "repcv5x10 — Nadeau-Bengio Corrected Test", [50, 100, 200, 500]),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for ax, data_sub, title, s_sizes in panels:
        ax.axhline(0.80, color="red", linestyle="--", linewidth=1.5, label="80% Target Power", alpha=0.85)
        ax.axvline(0.01, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(0.02, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        for idx, n in enumerate(s_sizes):
            sub_n = data_sub[data_sub["sample_size"] == n].sort_values(by="abs_gap")
            ax.plot(
                sub_n["abs_gap"], sub_n["power"],
                marker="o", linewidth=1.8, markersize=5,
                color=colors[idx], label=f"n = {n}"
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("True AUC Gap |Δ|", fontsize=10)
        ax.set_ylabel("Empirical Statistical Power", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=8, loc="lower right")

    plt.suptitle("CSIT332 Week 3: Tasks 3 & 4 — Statistical Power vs. True AUC Gap (The First Calculator)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
