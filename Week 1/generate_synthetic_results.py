"""
Generate Synthetic Experimental Results (600 Rows) for Component C (Group 5)
CSIT332 Principles of Machine Learning Semester Project

Specification:
- dataset_id: toy
- protocol_id: cv5
- n_sub: 100 and 500
- replicates: 20 per n_sub (replicate_id 1..20)
- classifiers: c1 (true_auc=0.85), c2 (true_auc=0.82), c3 (true_auc=0.76)
- folds: 1..5
- Total rows: 2 (n_sub) * 20 (reps) * 3 (clfs) * 5 (folds) = 600 rows
- Sampling variability:
    - n_sub = 100: sigma = 0.040 (higher variability)
    - n_sub = 500: sigma = 0.018 (lower variability)
- Fixed random seed: 42 (fully reproducible)
- Values clipped to [0.0, 1.0] and rounded to 4 decimal places.
"""

import os
import sys
import numpy as np
import pandas as pd


def _resolve_filepath(path: str) -> str:
    """Resolve file path relative to current working directory or script directory."""
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path


def generate_synthetic_results(
    truth_path: str = "truth.csv",
    output_path: str = "results.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate 600-row synthetic results dataset with principled sampling variability.
    """
    truth_path = _resolve_filepath(truth_path)
    output_path = _resolve_filepath(output_path)

    # Load ground truth values
    truth_df = pd.read_csv(truth_path)
    true_auc_map = dict(zip(truth_df["classifier_id"], truth_df["true_auc"]))

    # Parameters
    dataset_id = "toy"
    protocol_id = "cv5"
    sample_sizes = [100, 500]
    n_replicates = 20
    classifiers = ["c1", "c2", "c3"]
    n_folds = 5

    # Principled sampling standard deviation per sample size
    # n_sub=500 has lower variability than n_sub=100 (inversely proportional to sqrt(n))
    sigma_map = {
        100: 0.040,
        500: 0.018,
    }

    # Set random seed for full reproducibility
    rng = np.random.default_rng(seed)

    records = []

    for n_sub in sample_sizes:
        sigma = sigma_map[n_sub]
        for rep_id in range(1, n_replicates + 1):
            # To simulate realistic cross-validation sampling variability on a finite sample,
            # each replicate has a small sample-level shift, plus fold-level variation:
            rep_shift = rng.normal(loc=0.0, scale=sigma * 0.4)

            for clf_id in classifiers:
                base_auc = true_auc_map[clf_id]
                for fold_id in range(1, n_folds + 1):
                    fold_noise = rng.normal(loc=0.0, scale=sigma * 0.9)
                    simulated_auc = base_auc + rep_shift + fold_noise

                    # Clip to valid probability/AUC range [0.0, 1.0]
                    simulated_auc = float(np.clip(simulated_auc, 0.0, 1.0))
                    simulated_auc = round(simulated_auc, 4)

                    records.append({
                        "dataset_id": dataset_id,
                        "protocol_id": protocol_id,
                        "n_sub": n_sub,
                        "replicate_id": rep_id,
                        "classifier_id": clf_id,
                        "fold_id": fold_id,
                        "auc": simulated_auc,
                    })

    df = pd.DataFrame(records)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved {len(df)} rows to '{output_path}'.")

    return df


def validate_generated_dataset(
    df: pd.DataFrame, output_path: str = "results.csv"
):
    """
    Validate all 10 integrity requirements on the generated synthetic dataset.
    """
    print("\n" + "=" * 78)
    print("VALIDATING SYNTHETIC RESULTS DATASET")
    print("=" * 78)

    checks = []

    # 1. Total row count == 600
    total_rows = len(df)
    c1 = (total_rows == 600)
    checks.append(("Total row count == 600", c1, f"Found {total_rows} rows"))

    # 2. Number of replicate groups == 40
    rep_groups = df.groupby(["dataset_id", "protocol_id", "n_sub", "replicate_id"]).ngroups
    c2 = (rep_groups == 40)
    checks.append(("Total replicate groups == 40", c2, f"Found {rep_groups} groups"))

    # 3. 20 replicates at n_sub=100
    reps_100 = df[df["n_sub"] == 100]["replicate_id"].nunique()
    c3 = (reps_100 == 20)
    checks.append(("Replicates count for n_sub=100 == 20", c3, f"Found {reps_100}"))

    # 4. 20 replicates at n_sub=500
    reps_500 = df[df["n_sub"] == 500]["replicate_id"].nunique()
    c4 = (reps_500 == 20)
    checks.append(("Replicates count for n_sub=500 == 20", c4, f"Found {reps_500}"))

    # 5. Exactly 3 classifiers (c1, c2, c3)
    clfs = sorted(df["classifier_id"].unique().tolist())
    c5 = (clfs == ["c1", "c2", "c3"])
    checks.append(("Classifiers set == ['c1', 'c2', 'c3']", c5, f"Found {clfs}"))

    # 6. Exactly 5 folds per classifier per replicate
    fold_counts = df.groupby(["n_sub", "replicate_id", "classifier_id"])["fold_id"].count()
    c6 = bool((fold_counts == 5).all())
    checks.append(("Exactly 5 folds per classifier per replicate", c6, f"Min={fold_counts.min()}, Max={fold_counts.max()}"))

    # 7. Fold IDs are exactly {1, 2, 3, 4, 5}
    fold_id_sets = df.groupby(["n_sub", "replicate_id", "classifier_id"])["fold_id"].apply(lambda s: set(s))
    expected_set = {1, 2, 3, 4, 5}
    c7 = bool((fold_id_sets.apply(lambda s: s == expected_set)).all())
    checks.append(("Fold IDs are exactly {1, 2, 3, 4, 5}", c7, "All matched {1, 2, 3, 4, 5}"))

    # 8. No missing AUCs
    missing_aucs = df["auc"].isna().sum()
    c8 = (missing_aucs == 0)
    checks.append(("No missing / NaN AUC values", c8, f"Missing count: {missing_aucs}"))

    # 9. All AUCs between 0 and 1
    min_auc, max_auc = df["auc"].min(), df["auc"].max()
    c9 = (0.0 <= min_auc and max_auc <= 1.0)
    checks.append(("All AUCs in range [0, 1]", c9, f"Min={min_auc:.4f}, Max={max_auc:.4f}"))

    # 10. Empirical standard deviations (n_sub=500 must be lower than n_sub=100)
    # Calculate residual std around true_auc per n_sub
    truth_map = {"c1": 0.85, "c2": 0.82, "c3": 0.76}
    df_calc = df.copy()
    df_calc["true_auc"] = df_calc["classifier_id"].map(truth_map)
    df_calc["residual"] = df_calc["auc"] - df_calc["true_auc"]

    std_100 = float(df_calc[df_calc["n_sub"] == 100]["residual"].std())
    std_500 = float(df_calc[df_calc["n_sub"] == 500]["residual"].std())
    c10 = (std_500 < std_100)
    checks.append((
        f"Empirical std(n=500) [{std_500:.5f}] < std(n=100) [{std_100:.5f}]",
        c10,
        f"Ratio std(500)/std(100) = {std_500 / std_100:.3f}"
    ))

    # Print check summary
    all_passed = True
    for name, status, detail in checks:
        status_str = "PASS" if status else "FAIL"
        print(f"  {name:<60}: [{status_str}] ({detail})")
        if not status:
            all_passed = False

    print("=" * 78)
    if all_passed:
        print("[ALL 10 SYNTHETIC DATASET VALIDATIONS PASSED!]")
    else:
        print("[DATASET VALIDATION FAILED!]")
        sys.exit(1)

    print("\n--- Empirical Standard Deviation Summary ---")
    print(f"  n_sub = 100 empirical std: {std_100:.5f}")
    print(f"  n_sub = 500 empirical std: {std_500:.5f}")
    print(f"  Reduction in variability : {((std_100 - std_500) / std_100) * 100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    generated_df = generate_synthetic_results("truth.csv", "results.csv", seed=42)
    validate_generated_dataset(generated_df, "results.csv")
