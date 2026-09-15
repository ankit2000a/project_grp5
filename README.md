# Component C — Statistical Layer

## Project Context

Component C implements and validates the core statistical evaluation layer for **Group 5** of the **CSIT332 Principles of Machine Learning** semester project. 

The primary objective of Component C is to compute four benchmark performance metrics across multi-fold cross-validation replicates, compare observed classifier performance against known ground-truth values, and assess statistical power and optimism across varying finite sample sizes ($N=100$ vs $N=500$).

---

## Statistics Implemented

Component C implements the following four statistical metrics computed across experimental replicates:

1. **Winner-Correct Rate**:
   * For each replicate, compute each classifier's mean AUC across its five CV folds.
   * The classifier with the highest mean observed AUC is the declared winner.
   * Compares the declared winner against the genuinely best classifier from ground truth (`truth.csv`).
   * Winner-correct rate is the proportion of valid replicates where the declared winner equals the true best classifier.

2. **Mean Kendall's Tau ($\tau$)**:
   * Ranks classifiers by their replicate-level mean observed AUC and by their true AUC.
   * Computes Kendall's rank correlation coefficient ($\tau$) between the observed and true rankings using `scipy.stats.kendalltau`.
   * Mean Kendall's tau is the average of replicate-level $\tau$ values across valid replicates.

3. **Mean Optimism**:
   * Identifies the **declared winner** from observed mean AUC for each replicate.
   * Calculates: $\text{Optimism} = \text{Mean Observed AUC}(\text{Declared Winner}) - \text{True AUC}(\text{Declared Winner})$.
   * Measures the selection bias incurred when declaring the empirical winner.

4. **Power for $c_1$ vs $c_2$ (Paired $t$-Test)**:
   * Compares the genuinely superior classifier $c_1$ (`true_auc=0.85`) against $c_2$ (`true_auc=0.82`).
   * Pairs corresponding fold AUCs (fold 1 with fold 1, fold 2 with fold 2, etc.) and performs a paired $t$-test using `scipy.stats.ttest_rel`.
   * **Correct Detection** (Power): $p < 0.05$ and mean difference favors $c_1$ ($\text{mean}(c_1) > \text{mean}(c_2)$).
   * **Sign Error**: $p < 0.05$ and mean difference favors $c_2$ ($\text{mean}(c_2) > \text{mean}(c_1)$).
   * **No Detection**: $p \ge 0.05$ or mean difference equals zero.
   * Power is the proportion of valid replicates yielding a correct detection. Sign-error rate is reported alongside power.

---

## Data Files

The repository contains three primary CSV data files:

* **`truth.csv`**: Ground-truth benchmark table defining the true population AUC for each classifier:
  * $c_1 = 0.85$
  * $c_2 = 0.82$
  * $c_3 = 0.76$
* **`toy_results.csv`**: Baseline 15-row single-replicate dataset ($N=100$, 5 folds, 3 classifiers) used to verify hand-calculated assignment values.
* **`results.csv`**: 600-row synthetic experimental dataset simulating realistic finite-sample cross-validation results across:
  * 2 Sample sizes: $n\_sub = 100$ and $n\_sub = 500$
  * 20 Replicates per sample size (replicate_id 1..20)
  * 3 Classifiers ($c_1, c_2, c_3$)
  * 5 Folds per classifier per replicate (fold_id 1..5)
  * Total: $2 \times 20 \times 3 \times 5 = 600$ rows.

---

## Edge-Case Rules

Component C strictly enforces the following 8 pre-specified edge-case and ambiguity rules:

1. **Rule 1 (Exact Tie in Mean AUC)**: If two or more classifiers tie for the highest observed mean AUC, select the classifier with the lexicographically smallest `classifier_id` (e.g. $c_1$ beats $c_2$).
2. **Rule 2 (Non-Computable Fold AUC)**: Non-computable fold AUCs are treated as invalid and are never replaced with 0 or artificial values.
3. **Rule 3 (Incomplete Replicate & Fold ID Integrity)**: Every required classifier must have exactly 5 valid fold rows with integer fold IDs matching the exact set $\{1, 2, 3, 4, 5\}$. Duplicates, non-integers, missing folds, or out-of-range IDs invalidate the replicate.
4. **Rule 4 (Missing True AUC & Extra Classifiers)**: Ground truth is matched on both `dataset_id` and `classifier_id`. Every classifier appearing in results must have a valid numerical `true_auc` in `truth.csv`; missing entries or extra classifiers invalidate the replicate.
5. **Rule 5 (Missing / Non-Numeric / NaN AUC)**: Missing or NaN fold AUCs invalidate the affected fold and exclude the replicate if fewer than 5 valid folds remain.
6. **Rule 6 (Paired $t$-Test Cannot Be Performed)**: If paired folds cannot be matched, fold count $< 2$, diffs have zero variance, or $t$-test produces `NaN`/`inf`, the replicate is classified as **NO DETECTION** (`power=0`, `sign_error=0`).
7. **Rule 7 ($p = 0.05$ Threshold)**: Detection strictly requires $p < 0.05$; $p \ge 0.05$ is classified as **NO DETECTION**.
8. **Rule 8 (Mean Paired Difference = 0)**: If the mean paired difference is exactly zero, neither classifier is favored $\implies$ **NO DETECTION** (`power=0`, `sign_error=0`).

---

## Verification

The pipeline has been thoroughly verified:
* **10 Synthetic Dataset Validation Checks**: Passed (`generate_synthetic_results.py`).
* **13 Automated Unit Tests**: Passed (`component_c.py`), covering all 8 edge cases and sample-size independence.
* **Toy Hand-Calculation Verification**: Passed (`toy_results.csv`), exactly reproducing theoretical values.

---

## Results

### 1. Toy Dataset Verification (`toy_results.csv`)

| Metric | Calculated Value | Expected Value | Test Status |
| :--- | :--- | :--- | :--- |
| **Winner-correct rate** | `0.0000` | `0` | **PASS** |
| **Mean Kendall's tau** | `0.333333` ($1/3$) | `1/3` | **PASS** |
| **Mean optimism** | `0.010000` | `0.0100` | **PASS** |
| **Power ($c_1$ vs $c_2$)** | `0.0000` | `0` | **PASS** |
| **Sign-error rate ($c_1$ vs $c_2$)** | `0.0000` | `0` | **PASS** |

---

### 2. Experimental Dataset Results (`results.csv` by Sample Size)

```text
FINAL STATISTICAL SUMMARY BY SAMPLE SIZE

[SAMPLE SIZE: n_sub = 100]
---------------------------------------------
  Valid Replicates Analyzed   : 20
  Invalid Replicates Excluded : 0
  1. Winner-correct rate      : 0.9500
  2. Mean Kendall's tau       : 0.966667
  3. Mean optimism            : -0.003383
  4. Power (c1 vs c2)         : 0.1500
     Sign-error rate          : 0.0000
---------------------------------------------

[SAMPLE SIZE: n_sub = 500]
---------------------------------------------
  Valid Replicates Analyzed   : 20
  Invalid Replicates Excluded : 0
  1. Winner-correct rate      : 1.0000
  2. Mean Kendall's tau       : 1.000000
  3. Mean optimism            : 0.001238
  4. Power (c1 vs c2)         : 0.7500
     Sign-error rate          : 0.0000
---------------------------------------------
```

---

## How to Run

### 1. Generate & Validate Synthetic Dataset
```bash
python3 generate_synthetic_results.py
```

### 2. Run Statistical Evaluation & Automated Test Suite
```bash
python3 component_c.py
```

---

## Dependencies

* Python 3.8+
* `numpy`
* `pandas`
* `scipy`

Install dependencies via:
```bash
pip install numpy pandas scipy
```

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `component_c.py` | Core statistical calculations, sample-size evaluation, 8 edge-case rules, and 13 unit tests. |
| `generate_synthetic_results.py` | Reproducible generator and validator for the 600-row synthetic results dataset. |
| `truth.csv` | Ground truth classifier AUC benchmark values ($c_1=0.85, c_2=0.82, c_3=0.76$). |
| `toy_results.csv` | 15-row assignment toy test dataset used for hand-calculation verification. |
| `results.csv` | 600-row experimental 5-fold CV results across 40 replicates ($N=100$ and $N=500$). |
| `README.md` | Comprehensive documentation and verification summary for Component C. |
| `.gitignore` | Standard Git ignore configuration for Python, macOS, and IDE artifacts. |
