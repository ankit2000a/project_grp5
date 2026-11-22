"""
CSIT332 Week 3 — Group 5 — Component C: Task 2 — Candidate-Count Analysis
Investigates whether trying more models helps by evaluating all C(6, k) subsets
for k in {2, 3, 4, 5, 6} (15, 20, 15, 6, 1 subsets) on CV5.
Computes winner-correct rate beside 1/k chance baseline and measures selection optimism growth.
"""

import os
import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .component_c import (
        CLASSIFIER_IDS,
        SAMPLE_SIZES,
        get_truth_subset,
        calculate_replicate_means,
    )
except ImportError:
    from component_c import (
        CLASSIFIER_IDS,
        SAMPLE_SIZES,
        get_truth_subset,
        calculate_replicate_means,
    )


def run_candidate_count_analysis(
    results_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    dataset_id: str = "synthetic_pilot",
    protocol_id: str = "cv5",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Candidate-Count Analysis (Task 2) on CV5:
    For k in {2, 3, 4, 5, 6}, evaluates every subset of k classifiers:
    - k=2: 15 subsets
    - k=3: 20 subsets
    - k=4: 15 subsets
    - k=5: 6 subsets
    - k=6: 1 subset
    
    Within each subset:
    - Computes declared winner via highest replicate mean AUC (lexicographical tie break).
    - Computes ground-truth best in that subset (truth-at-n, handling ties).
    - Measures winner-correct indicator and selection optimism.
    - Averages across all replicates and across all subsets of size k.
    
    Returns:
        (summary_df, detailed_subsets_df)
    """
    proto_df = results_df[
        (results_df["dataset_id"] == dataset_id) & (results_df["protocol_id"] == protocol_id)
    ]
    if proto_df.empty:
        raise ValueError(f"No results found for dataset '{dataset_id}' and protocol '{protocol_id}'.")

    # Pre-calculate replicate-level mean AUCs for all replicates
    rep_means = calculate_replicate_means(proto_df)

    summary_rows = []
    detail_rows = []

    for n in SAMPLE_SIZES:
        sub_reps = rep_means[rep_means["n_sub"] == n]
        if sub_reps.empty:
            continue

        truth_sub = get_truth_subset(truth_df, dataset_id=dataset_id, n_sub=n, mode="per_sample")
        if truth_sub.empty:
            continue
        truth_dict = truth_sub.set_index("classifier_id")["true_auc"].to_dict()

        replicate_ids = sorted(sub_reps["replicate_id"].unique().tolist())
        num_reps = len(replicate_ids)

        for k in [2, 3, 4, 5, 6]:
            subsets = list(itertools.combinations(CLASSIFIER_IDS, k))
            expected_count = {2: 15, 3: 20, 4: 15, 5: 6, 6: 1}[k]
            assert len(subsets) == expected_count, f"Expected {expected_count} subsets for k={k}, got {len(subsets)}"

            k_wc_means = []
            k_opt_means = []

            for subset in subsets:
                subset_clfs = list(subset)
                subset_str = "+".join(subset_clfs)

                # Find true best in this subset
                max_true_auc = max(truth_dict[c] for c in subset_clfs)
                true_winners = {c for c in subset_clfs if np.isclose(truth_dict[c], max_true_auc, atol=1e-9)}

                # Filter replicates to subset classifiers
                subset_rep_data = sub_reps[sub_reps["classifier_id"].isin(subset_clfs)]

                # For each replicate, find declared winner with lexicographical tie-break
                sorted_rep = subset_rep_data.sort_values(
                    by=["mean_auc", "classifier_id"], ascending=[False, True]
                )
                rep_winners = sorted_rep.groupby("replicate_id").first().reset_index()

                # Winner-correct: declared winner is in true_winners
                is_correct = rep_winners["classifier_id"].isin(true_winners).astype(float)
                wc_rate = float(is_correct.mean())

                # Optimism: observed mean AUC - true AUC of that winner
                winner_true_aucs = rep_winners["classifier_id"].map(truth_dict)
                rep_optimisms = rep_winners["mean_auc"] - winner_true_aucs
                opt_mean = float(rep_optimisms.mean())

                k_wc_means.append(wc_rate)
                k_opt_means.append(opt_mean)

                detail_rows.append({
                    "n_sub": n,
                    "k": k,
                    "subset": subset_str,
                    "winner_correct": wc_rate,
                    "optimism": opt_mean,
                    "num_replicates": num_reps,
                })

            summary_rows.append({
                "n_sub": n,
                "k": k,
                "num_subsets": len(subsets),
                "chance_baseline": 1.0 / k,
                "winner_correct_mean": float(np.mean(k_wc_means)),
                "winner_correct_std": float(np.std(k_wc_means, ddof=1)) if len(k_wc_means) > 1 else 0.0,
                "optimism_mean": float(np.mean(k_opt_means)),
                "optimism_std": float(np.std(k_opt_means, ddof=1)) if len(k_opt_means) > 1 else 0.0,
            })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detail_rows)
    return summary_df, detailed_df


def plot_candidate_analysis(
    summary_df: pd.DataFrame,
    output_path: str = "plots/candidate_count_optimism.png",
):
    """
    Generate high-resolution publication plot for Task 2:
    - Left Panel: Winner-Correct Rate vs k alongside 1/k chance baseline.
    - Right Panel: Selection Optimism vs k showing growth with candidate count.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    sample_sizes = sorted(summary_df["n_sub"].unique().tolist())
    k_vals = sorted(summary_df["k"].unique().tolist())
    chance_rates = [1.0 / k for k in k_vals]

    # Panel A: Winner-Correct Rate vs k
    ax_wc = axes[0]
    ax_wc.plot(k_vals, chance_rates, "k--", linewidth=2.0, label="Chance Baseline (1/k)", alpha=0.85)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_sizes)))
    for idx, n in enumerate(sample_sizes):
        sub = summary_df[summary_df["n_sub"] == n].sort_values(by="k")
        ax_wc.plot(
            sub["k"], sub["winner_correct_mean"],
            marker="o", linewidth=2.0, color=colors[idx],
            label=f"n = {n}"
        )

    ax_wc.set_title("Winner-Correct Rate vs. Candidate Count (k)", fontsize=13, fontweight="bold")
    ax_wc.set_xlabel("Number of Candidate Classifiers (k)", fontsize=11)
    ax_wc.set_ylabel("Mean Winner-Correct Rate", fontsize=11)
    ax_wc.set_xticks(k_vals)
    ax_wc.set_ylim(0.0, 1.05)
    ax_wc.grid(True, linestyle=":", alpha=0.6)
    ax_wc.legend(frameon=True, fontsize=9)

    # Panel B: Selection Optimism vs k
    ax_opt = axes[1]
    for idx, n in enumerate(sample_sizes):
        sub = summary_df[summary_df["n_sub"] == n].sort_values(by="k")
        ax_opt.plot(
            sub["k"], sub["optimism_mean"],
            marker="s", linewidth=2.0, color=colors[idx],
            label=f"n = {n}"
        )

    ax_opt.set_title("Selection Optimism vs. Candidate Count (k)", fontsize=13, fontweight="bold")
    ax_opt.set_xlabel("Number of Candidate Classifiers (k)", fontsize=11)
    ax_opt.set_ylabel("Mean Selection Optimism (Observed - True AUC)", fontsize=11)
    ax_opt.set_xticks(k_vals)
    ax_opt.grid(True, linestyle=":", alpha=0.6)
    ax_opt.legend(frameon=True, fontsize=9)

    plt.suptitle("CSIT332 Week 3: Task 2 — Does Trying More Models Help?", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
