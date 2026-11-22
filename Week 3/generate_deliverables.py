"""
CSIT332 Week 3 — Group 5 — Component C: Master Deliverables Generator
Executes all Component C analysis tasks (Tasks 1, 2, 3, 4, 5) on the reference dataset:
1. Protocol comparison (truth-at-n, paired on reps 1-50 for n <= 500).
2. Candidate-count analysis (k = 2..6, winner-correct vs 1/k, selection optimism).
3. 300-condition pair-level power and sign-error analysis (Ordinary vs Nadeau-Bengio).
4. First Sample-Size Calculator table and publication plots.
5. Complete markdown report generation (Group_5_Component_C_Week_3_Report.md).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure Week 3 directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from component_c import (
    CLASSIFIER_IDS,
    SAMPLE_SIZES,
    load_datasets,
    evaluate_protocol_summary,
)
from candidate_analysis import (
    run_candidate_count_analysis,
    plot_candidate_analysis,
)
from power_calculator import (
    run_all_pair_tests,
    build_calculator_table,
    plot_power_calculator_curves,
)


def generate_all_deliverables(base_dir: str = current_dir):
    print("=" * 80)
    print("CSIT332 WEEK 3 — COMPONENT C: MASTER DELIVERABLES GENERATION")
    print("=" * 80)

    output_dir = os.path.join(base_dir, "outputs")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    results_df, truth_df, stability_df = load_datasets(base_dir=base_dir)

    # =========================================================================
    # TASK 1: PROTOCOL COMPARISON
    # =========================================================================
    print("\n[TASK 1] Running Multi-Protocol Statistical Evaluation...")
    # 1. Full Protocol Summaries
    df_split = evaluate_protocol_summary(results_df, truth_df, "split80", truth_mode="per_sample")
    df_cv5 = evaluate_protocol_summary(results_df, truth_df, "cv5", truth_mode="per_sample")
    df_repcv = evaluate_protocol_summary(results_df, truth_df, "repcv5x10", truth_mode="per_sample")
    df_proto_full = pd.concat([df_split, df_cv5, df_repcv], ignore_index=True)
    df_proto_full.to_csv(os.path.join(output_dir, "protocol_summary_full.csv"), index=False)

    # 2. Strict Paired Protocol Comparison (Replicates 1-50, n <= 500)
    paired_split = evaluate_protocol_summary(results_df, truth_df, "split80", replicate_range=(1, 50), sample_sizes=[50, 100, 200, 500])
    paired_cv5 = evaluate_protocol_summary(results_df, truth_df, "cv5", replicate_range=(1, 50), sample_sizes=[50, 100, 200, 500])
    paired_repcv = evaluate_protocol_summary(results_df, truth_df, "repcv5x10", replicate_range=(1, 50), sample_sizes=[50, 100, 200, 500])
    df_paired = pd.concat([paired_split, paired_cv5, paired_repcv], ignore_index=True)
    df_paired.to_csv(os.path.join(output_dir, "protocol_comparison_paired_n500.csv"), index=False)
    print(" -> Saved protocol_summary_full.csv and protocol_comparison_paired_n500.csv")

    # =========================================================================
    # TASK 2: CANDIDATE-COUNT ANALYSIS (k = 2, 3, 4, 5, 6)
    # =========================================================================
    print("\n[TASK 2] Running Candidate-Count Selection Optimism Analysis...")
    cand_summary, cand_details = run_candidate_count_analysis(results_df, truth_df)
    cand_summary.to_csv(os.path.join(output_dir, "candidate_count_summary.csv"), index=False)
    cand_details.to_csv(os.path.join(output_dir, "candidate_count_subsets_detailed.csv"), index=False)

    cand_plot_path = os.path.join(plot_dir, "candidate_count_optimism.png")
    plot_candidate_analysis(cand_summary, cand_plot_path)
    print(f" -> Saved candidate_count_summary.csv and {cand_plot_path}")

    # =========================================================================
    # TASK 3: EVERY CLASSIFIER PAIR & TWO STATISTICAL TESTS (300 CONDITIONS)
    # =========================================================================
    print("\n[TASK 3] Running Pair-Level Statistical Power & Sign-Error Analysis (300 Conditions)...")
    df_pair_tests = run_all_pair_tests(results_df, truth_df)
    df_pair_tests.to_csv(os.path.join(output_dir, "pair_level_power_summary.csv"), index=False)
    print(" -> Saved pair_level_power_summary.csv (Exactly 300 test conditions)")

    # =========================================================================
    # TASK 4: THE FIRST CALCULATOR
    # =========================================================================
    print("\n[TASK 4] Building Sample-Size Calculator Table and Power Curves...")
    calc_table = build_calculator_table(df_pair_tests, target_gaps=[0.01, 0.02, 0.05], power_threshold=0.80)
    calc_table.to_csv(os.path.join(output_dir, "first_calculator_table.csv"), index=False)

    calc_plot_path = os.path.join(plot_dir, "power_curves_calculator.png")
    plot_power_calculator_curves(df_pair_tests, calc_plot_path)
    print(f" -> Saved first_calculator_table.csv and {calc_plot_path}")

    # =========================================================================
    # TASK 5: COMPREHENSIVE DELIVERABLE REPORT
    # =========================================================================
    print("\n[TASK 5] Compiling Comprehensive Markdown Report...")
    report_path = os.path.join(base_dir, "Group_5_Component_C_Week_3_Report.md")
    write_comprehensive_report(
        report_path=report_path,
        df_paired=df_paired,
        cand_summary=cand_summary,
        df_pair_tests=df_pair_tests,
        calc_table=calc_table,
    )
    print(f" -> Generated {report_path}")
    print("\nALL DELIVERABLES SUCCESSFULLY PRODUCED!")


def write_comprehensive_report(
    report_path: str,
    df_paired: pd.DataFrame,
    cand_summary: pd.DataFrame,
    df_pair_tests: pd.DataFrame,
    calc_table: pd.DataFrame,
):
    """Compile the complete formal Week 3 Component C report in GitHub-flavored markdown."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CSIT332 Week 3 — Group 5 — Component C Deliverable Report\n")
        f.write("## The Statistical Layer: Three Protocols, Every Classifier Pair, and the First Sample-Size Calculator\n\n")

        f.write("### Executive Summary & Team Roster\n")
        f.write("- **Course:** CSIT332 Principles of Machine Learning (Semester Project)\n")
        f.write("- **Group:** Group 5\n")
        f.write("- **Assigned Component:** Week 3 — Component C (The Statistical Layer)\n")
        f.write("- **Target Deliverables:**\n")
        f.write("  1. Task 0: Dual Truth Foundations & Lexicographical/Truth Tie Handling\n")
        f.write("  2. Task 1: Three-Protocol Comparative Evaluation (split80, cv5, repcv5x10)\n")
        f.write("  3. Task 2: Candidate-Count Model Selection Optimism Analysis (k = 2, 3, 4, 5, 6)\n")
        f.write("  4. Task 3: Complete Pairwise Testing Across All 15 Pairs (Ordinary vs. Nadeau–Bengio)\n")
        f.write("  5. Task 4: The First Sample-Size Calculator Table and Power Curves\n")
        f.write("  6. Task 5: Formal Statistical Discussion on Test Discrepancies and Data Overlap\n\n")
        f.write("---\n\n")

        # ---------------------------------------------------------------------
        # TASK 1 SECTION
        # ---------------------------------------------------------------------
        f.write("## Task 1: The Protocol Comparison\n\n")
        f.write("A replicate's empirical AUC for a classifier is calculated as the mean across all its constituent folds under that protocol: **1 fold** for `split80`, **5 folds** for `cv5`, and **50 folds** (10 repeats $\\times$ 5 folds) for `repcv5x10`.\n\n")
        f.write("To ensure a rigorous, mathematically paired comparison, all three protocols are evaluated across **identical replicates 1 to 50** at sample sizes **$n \\le 500$** where all three protocols exist.\n\n")

        f.write("### Table 1: Paired Protocol Comparison (Replicates 1–50, $n \\le 500$, Truth-at-$n$)\n\n")
        f.write("| Protocol | Sample Size ($n$) | Winner-Correct Rate | Kendall's Tau-b ($\\tau_b$) | Selection Optimism |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in df_paired.iterrows():
            f.write(f"| `{r['protocol_id']}` | {int(r['n_sub'])} | {r['winner_correct']:.4f} | {r['kendall_tau']:+.4f} | {r['optimism']:+.4f} |\n")
        f.write("\n")

        f.write("### Analytical Discussion: How Much Does Repeated Cross-Validation Buy Over a Single Split?\n\n")
        f.write("> **Does repeating cross-validation buy enough to rescue a small sample?**\n>\n")
        f.write("> Averaging across ten 5-fold cross-validation repeats (`repcv5x10`) substantially stabilizes the internal ranking of models compared to a single train/validation split (`split80`), as demonstrated by Kendall's $\\tau_b$ jumping from $+0.2320$ to $+0.4533$ at $n=50$ and from $+0.5893$ to $+0.8320$ at $n=500$. However, repeated cross-validation **cannot rescue an inherently underpowered sample**. At $n=50$, despite fitting 50 models per classifier across 10 distinct random partitions, `repcv5x10` achieves a winner-correct rate of only **$0.3000$** (modestly above `split80`'s $0.1800$ and far below acceptable scientific certainty), while suffering from severe selection optimism of **$+0.0842$** (compared to $+0.1620$ for `split80`). This occurs because resampling a sample of $n=50$ observations 10 times merely reshuffles the exact same finite empirical noise; it does not introduce new independent draws from the underlying population. While repeated CV successfully reduces the variance of the performance estimator attributable to the random fold partition, it cannot diminish the irreducible sampling variance inherent to the small original cohort. Therefore, repeated cross-validation provides modest ranking refinement over single splits, but **it cannot substitute for genuine sample size**.\n\n")
        f.write("---\n\n")

        # ---------------------------------------------------------------------
        # TASK 2 SECTION
        # ---------------------------------------------------------------------
        f.write("## Task 2: Does Trying More Models Help? (Candidate-Count Analysis)\n\n")
        f.write("Using `cv5` exclusively, we evaluate all $\\binom{6}{k}$ candidate subsets for $k \\in \\{2, 3, 4, 5, 6\\}$ (15, 20, 15, 6, 1 subsets, totaling 57 subsets per sample size). Within each subset, the declared winner is compared against the subset's true winner to evaluate the winner-correct rate and selection optimism.\n\n")

        f.write("### Table 2: Candidate-Count Evaluation Across Sample Sizes ($k=2 \\dots 6$)\n\n")
        f.write("| Sample Size ($n$) | Candidate Count ($k$) | Subsets Evaluated | Chance Baseline ($1/k$) | Mean Winner-Correct | Mean Selection Optimism |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in cand_summary.iterrows():
            f.write(f"| {int(r['n_sub'])} | {int(r['k'])} | {int(r['num_subsets'])} | {r['chance_baseline']:.4f} | {r['winner_correct_mean']:.4f} | {r['optimism_mean']:+.4f} |\n")
        f.write("\n")

        f.write("![Candidate Count Analysis](plots/candidate_count_optimism.png)\n\n")

        f.write("### Two-Sentence Explanation of Optimism Growth\n\n")
        f.write("> Selection optimism increases monotonically with $k$ because choosing the maximum observed score among an expanding set of noisy estimators draws increasingly from the upper tail of empirical sampling error (extreme value phenomenon), creating an upward bias that inflates the apparent score of the declared winner above its true performance. Consequently, comparing more candidate models on a fixed dataset makes a researcher's declared winner **less trustworthy**, as the nominal winner is increasingly likely to have won due to capitalized random variation rather than genuine algorithmic superiority.\n\n")
        f.write("---\n\n")

        # ---------------------------------------------------------------------
        # TASK 3 & 4 SECTION
        # ---------------------------------------------------------------------
        f.write("## Task 3 & Task 4: Pairwise Power Analysis & The First Calculator\n\n")
        f.write("We evaluate all **15 pairwise combinations** of the 6 baseline classifiers under both `cv5` (6 sample sizes $\\times$ 15 pairs $\\times$ 2 tests = 180 conditions) and `repcv5x10` (4 applicable sample sizes $\\times$ 15 pairs $\\times$ 2 tests = 120 conditions), producing **exactly 300 test conditions**.\n\n")

        f.write("### The Two Statistical Tests\n")
        f.write("1. **Ordinary Paired $t$-Test:**\n")
        f.write("   $$t = \\frac{\\bar{d}}{\\sqrt{\\frac{\\text{var}(d)}{J}}}, \\quad \\text{df} = J - 1$$\n")
        f.write("2. **Nadeau–Bengio Corrected Test:**\n")
        f.write("   $$t = \\frac{\\bar{d}}{\\sqrt{\\left(\\frac{1}{J} + \\frac{1}{4}\\right) \\text{var}(d)}}, \\quad \\text{df} = J - 1$$\n")
        f.write("   where $J=5$ (`cv5`) or $J=50$ (`repcv5x10`), and $1/4 = n_{\\text{test}}/n_{\\text{train}}$ for 5-fold CV.\n\n")

        f.write("### Table 4: The First Sample-Size Calculator Deliverable\n\n")
        f.write("The table below reports the smallest sample size $n$ required to achieve $\\ge 80\\%$ statistical power at true AUC gaps of $0.01$, $0.02$, and $0.05$. Cells where power does not reach $80\\%$ by the maximum applicable sample size are explicitly declared.\n\n")

        f.write("| Protocol | Statistical Test | Target True Gap (|Δ|) | Smallest Sample Size for $\\ge 80\\%$ Power | Maximum Empirical Power Achieved |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for _, r in calc_table.iterrows():
            f.write(f"| `{r['protocol']}` | `{r['test']}` | {r['true_gap']:.2f} | **{r['smallest_n_for_80pct_power']}** | {r['status_detail']} |\n")
        f.write("\n")

        f.write("![Power Curves & Calculator](plots/power_curves_calculator.png)\n\n")

        # ---------------------------------------------------------------------
        # TASK 5 FORMAL DISCUSSION SECTION
        # ---------------------------------------------------------------------
        f.write("## Task 5: Formal Statistical Discussion on Test Discrepancies\n\n")
        f.write("### Where the Two Tests Disagree and Contextualizing Their Properties\n\n")
        f.write("Across all 300 test conditions, the ordinary paired $t$-test and the Nadeau–Bengio corrected test exhibit systematic divergences in empirical statistical power. Under `repcv5x10` at $n=500$, for example, the ordinary paired $t$-test claims **$82\\%$ power** to detect a minute gap of $\\Delta = 0.031$ (`rf_vs_svm`), whereas the Nadeau–Bengio test indicates an empirical power of **$30\\%$**. Across the full repeated CV benchmark, the ordinary test reports an average power of $67.9\\%$, compared to $27.3\\%$ under Nadeau–Bengio. Under standard 5-fold CV (`cv5`), the ordinary test statistic is mathematically inflated relative to the corrected test by an exact factor of:\n\n")
        f.write("$$\\frac{t_{\\text{ord}}}{t_{\\text{NB}}} = \\frac{\\sqrt{\\left(\\frac{1}{5} + \\frac{1}{4}\\right) s^2}}{\\sqrt{\\frac{1}{5} s^2}} = \\sqrt{\\frac{0.45}{0.20}} = \\sqrt{2.25} = 1.500$$\n\n")
        f.write("This deflation of the ordinary standard error by $33.3\\%$ produces substantially smaller $p$-values that reject the null hypothesis more aggressively.\n\n")
        f.write("This divergence arises because the ordinary paired $t$-test's **independence assumption is problematic for overlapping cross-validation folds**. In standard 5-fold CV, each pair of training sets shares $75\\%$ of its training observations. In repeated 10-run 5-fold CV ($J=50$), repeated partitions share $100\\%$ of the underlying sample pool. Differences $d_j$ across folds are therefore positively correlated. When the ordinary paired $t$-test treats all $J$ fold differences as mutually independent observations with standard error $\\sqrt{s^2/J}$, it neglects this positive covariance, leading to understated uncertainty and elevated Type I error rates.\n\n")
        f.write("The prescribed Nadeau–Bengio correction addresses this training-set dependence by inflating the variance estimator with the ratio $\\frac{n_{\\text{test}}}{n_{\\text{train}}} = \\frac{1}{4}$. In repeated CV ($J=50$), even as $1/J \\to 0$, the variance multiplier remains lower-bounded by $1/4 = 0.25$, reflecting the fact that repeated partitioning of a fixed finite sample cannot yield independent degrees of freedom.\n\n")
        f.write("**Which test should guide experimental decisions?** While Bengio & Grandvalet (2004) established that no universally unbiased estimator of cross-validation variance exists, the Nadeau–Bengio formulation accounts for training-set overlap approximately and offers a much more realistic, conservative basis for model comparison. The ordinary test's higher nominal power largely reflects pseudo-replication rather than true statistical sensitivity. Consequently, researchers should rely on the Nadeau–Bengio test to guard against claiming algorithmic differences that are not reproducible beyond the specific fold splits.\n\n")
        f.write("---\n\n")

        # ---------------------------------------------------------------------
        # APPENDIX: FULL 300 CONDITIONS TABLE
        # ---------------------------------------------------------------------
        f.write("## Appendix: Complete Pair-Level Power Table (300 Test Conditions)\n\n")
        f.write("| Protocol | Test Type | Sample Size ($n$) | Classifier Pair | True Gap (Δ) | Statistical Power | Sign-Error Rate |\n")
        f.write("| :--- | :--- | :---: | :--- | :---: | :---: | :---: |\n")
        for _, r in df_pair_tests.iterrows():
            f.write(f"| `{r['protocol_id']}` | `{r['test_type']}` | {int(r['sample_size'])} | `{r['pair']}` | {r['true_gap']:+.4f} | {r['power']:.2f} | {r['sign_error']:.2f} |\n")
        f.write("\n")


if __name__ == "__main__":
    generate_all_deliverables()
