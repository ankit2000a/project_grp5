# Component C — The Statistical Evaluation Layer
## CSIT332: Principles of Machine Learning · Semester Project

**Group:** Group 5  
**Component:** Component C (Statistical Layer & Model Evaluation Benchmark)  
**Academic Term:** Fifth Semester (Fall 2026)  

---

## 1. Project Overview

Component C implements, verifies, and analyzes the core statistical evaluation layer for the machine learning benchmark project. The objective is to evaluate multi-model comparative study designs under finite-sample cross-validation, measuring:
1. **Winner-Correct Rate:** The probability that the empirically selected top model is genuinely the best.
2. **Kendall's Tau ($\tau_b$):** The rank correlation between cross-validated rankings and ground-truth generalization order.
3. **Mean Optimism:** The expected inflation in the winner's cross-validated score due to selection bias.
4. **Paired $t$-Test Power & Sign-Error Rate:** The sensitivity and directional reliability of comparative hypotheses under varying sample and effect sizes.

---

## 2. Week 2: Benchmark Evaluation on Reference Data & The Dual Nature of Truth

### 2.1 The Problem & Methodological Correction
During preliminary runs on the 12,000-row reference machinery, evaluating models on 50 samples against abundant-data (asymptotic) performance produced a **negative optimism value of −0.035**. 

Optimism is intended to measure the upward inflation from selecting the maximum of noisy estimators ($> 0$). It appeared negative because two opposing phenomena were mixed together:
* **Selection Bias (Optimism):** Capitalizing on chance pushes the winner's score *upward*.
* **Finite-Sample Skill Deficit:** A classifier trained on 40 rows lacks sufficient data to reach its asymptotic performance, pushing its score *downward*.

At small sample sizes, the deficit swamped the selection bias ($+0.106 - 0.141 = -0.035$).

**The Week 2 Correction:** Ground truth is measured at **each sample size** ($n \in \{50, 100, 200, 500\}$) by evaluating models trained on the exact cross-validation training folds against a permanent frozen test set of 6,000 independent rows (**Version A: Truth at $n$**), alongside the abundant-data benchmark (**Version B: Asymptotic Truth**).

* **Version A** isolates pure selection bias: optimism is strictly positive ($+0.106 \to +0.022$) and shrinks as $n$ grows.
* **Version B** penalizes models for finite sample size and misidentifies small-sample winners because algorithms have different learning curves (e.g. `logreg` is optimal on 50 rows, while `gb` only dominates with 12,600 rows).

---

### 2.2 Table 1: Four Quantities × Four Sample Sizes (Version A vs. Version B)

Evaluated across all 100 replicates (12,000 rows) of `Week 2/reference_results.csv` against `Week 2/reference_truth.csv`:

| Sample Size ($n$) | Winner-Correct Rate (Ver A / Ver B) | Mean Kendall $\tau_b$ (Ver A / Ver B)* | Mean Optimism (Ver A / Ver B) | True Best Model at $n$ [True AUC] | Empirical Winner Distribution (100 Replicates) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$n = 50$** | **0.4300** / 0.1500 | **+0.3166** / +0.0177 | **+0.1059** / −0.0352 | `logreg` [0.6903] | `logreg` 43%, `gb` 15%, `svm` 13%, `rf` 13%, `knn` 12%, `dtree` 4% |
| **$n = 100$** | **0.4100** / 0.1200 | **+0.5091** / +0.2008 | **+0.0559** / −0.0439 | `logreg` [0.7232] | `logreg` 41%, `svm` 32%, `gb` 12%, `rf` 10%, `knn` 5% |
| **$n = 200$** | **0.4700** / 0.0600 | **+0.6908** / +0.3358 | **+0.0406** / −0.0216 | `svm` [0.7544] | `svm` 47%, `logreg` 32%, `rf` 13%, `gb` 6%, `dtree` 1%, `knn` 1% |
| **$n = 500$** | **0.6900** / 0.0800 | **+0.8063** / +0.4728 | **+0.0220** / −0.0091 | `svm` [0.7856] | `svm` 69%, `logreg` 23%, `gb` 8% |

*\*Note: Version A Kendall's $\tau_b$ is the primary finite-sample rank correlation. Version B is included as supplementary context to illustrate the divergence in ordinal rankings. In Version B, the asymptotic target winner is `gb` (true AUC = 0.836803).*

* **Winner-Correct Rate:** Generally improves as sample size increases, rising from 43% at $n = 50$ to 69% at $n = 500$, dipping slightly to 41% at $n = 100$ before increasing at the larger sample sizes.
* **Optimism:** Strictly positive and monotonic under Version A ($+0.1059 \to +0.0559 \to +0.0406 \to +0.0220$), perfectly matching statistical theory.

---

### 2.3 Table 2: Paired $t$-Test Power Analysis (Close vs. Wide Pairs)

Evaluated via 5-fold paired $t$-tests ($\alpha = 0.05$) across all 100 replicates:
* **Close Pair:** `gb` vs `knn` (near-tie: true $\Delta = +0.0016 < 0.01$ at $n=50$)
* **Wide Pair:** `logreg` vs `dtree` (wide gap: true $\Delta > 0.12$ across all $n$)

| Classifier Pair | Sample Size ($n$) | True Gap at $n$ ($\Delta$ AUC) | Observed Mean $\Delta$ | Power (Ver A) [Correct Detection] | Sign-Error (Ver A) [Wrong Direction] | Power (Ver B)* [Asymptotic Ref] | Sign-Error (Ver B)* [Asymptotic Ref] | Methodological Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`gb` vs `knn`**<br/>*(Close Pair: $\Delta < 0.01$ at $n=50$)* | 50<br/>100<br/>200<br/>500 | +0.0016<br/>+0.0183<br/>+0.0406<br/>+0.0634 | +0.0034<br/>+0.0263<br/>+0.0383<br/>+0.0667 | 0.0500 (5%)<br/>0.0800 (8%)<br/>0.1400 (14%)<br/>0.3800 (38%) | **0.0400 (4%)**<br/>0.0100 (1%)<br/>0.0000 (0%)<br/>0.0000 (0%) | 0.0500<br/>0.0800<br/>0.1400<br/>0.3800 | 0.0400<br/>0.0100<br/>0.0000<br/>0.0000 | At $n=50$, true difference is negligible. Statistical power is at nominal $\alpha$ (5%), while sign error is 4%. A significant test is a coin flip. |
| **`logreg` vs `dtree`**<br/>*(Wide Pair: $\Delta > 0.12$ across all $n$)* | 50<br/>100<br/>200<br/>500 | +0.1290<br/>+0.1494<br/>+0.1558<br/>+0.1415 | +0.1342<br/>+0.1504<br/>+0.1682<br/>+0.1469 | 0.1800 (18%)<br/>0.4200 (42%)<br/>0.5200 (52%)<br/>**0.8700 (87%)** | 0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%) | 0.1800<br/>0.4200<br/>0.5200<br/>0.8700 | 0.0000<br/>0.0000<br/>0.0000<br/>0.0000 | Large gap completely prevents sign errors (0%). Power scales rapidly from 18% to 87%, achieving standard clinical power ($\ge 80\%$) at $n=500$. |

*\*Note: For gb vs knn and logreg vs dtree, both Version A and Version B agree on the superior classifier direction (gb > knn and logreg > dtree), so their numerical rates match.*

**Supplementary Reversal Analysis (`svm` vs `logreg`):**  
At $n=50$, `logreg` is genuinely superior by $+0.0981$. Version A credits favoring `logreg` ($\text{Power} = 3\%$, $\text{Sign Error} = 1\%$). Version B treats `svm` as asymptotic best, inverting the verdict and misclassifying favoring `logreg` as a Sign Error ($3\%$) and favoring `svm` as Power ($1\%$)! At $n=200$, a near-tie occurs ($\Delta = +0.0021$ favoring `svm`), with $\text{Power} = 3\%$ and $\text{Sign Error} = 1\%$.

---

### 2.4 Week 2 Deliverables & Files
* **Core Code:** [`Week 2/component_c.py`](Week%202/component_c.py) (includes 18 automated unit tests).
* **Deliverable PDF:** [`Week 2/Group_5_Component_C_Week_2_Report.pdf`](Week%202/Group_5_Component_C_Week_2_Report.pdf) (formatted for LMS submission).
* **Deliverable Markdown:** [`Week 2/Group_5_Component_C_Week_2_Report.md`](Week%202/Group_5_Component_C_Week_2_Report.md) (full text and explanation).
* **CSV Exports:** [`Week 2/four_quantities_summary.csv`](Week%202/four_quantities_summary.csv), [`Week 2/power_analysis_summary.csv`](Week%202/power_analysis_summary.csv).
* **PDF Generator:** [`Week 2/generate_pdf_report.py`](Week%202/generate_pdf_report.py).

---

## 3. Week 1: Synthetic Pilot Baseline & Verification

### 3.1 Overview
In Week 1, Component C established the statistical architecture, edge-case rules, and truth-tie logic, verified against hand-calculated toy data and a 600-row synthetic pilot dataset ($N=100$ and $N=500$).

### 3.2 Hand-Calculation Toy Verification (`Week 1/toy_results.csv`)
* **Winner-Correct Rate:** `0.0000` (Expected `0`) $\implies$ **PASS**
* **Mean Kendall's Tau:** `0.333333` ($1/3$) (Expected $1/3$) $\implies$ **PASS**
* **Mean Optimism:** `0.0100` (Expected `0.0100`) $\implies$ **PASS**
* **Power ($c_1$ vs $c_2$):** `0.0000` (Expected `0`) $\implies$ **PASS**
* **Sign-Error Rate:** `0.0000` (Expected `0`) $\implies$ **PASS**

### 3.3 Week 1 Deliverables & Files
* **Code:** [`Week 1/component_c.py`](Week%201/component_c.py), [`Week 1/generate_synthetic_results.py`](Week%201/generate_synthetic_results.py).
* **Data:** [`Week 1/truth.csv`](Week%201/truth.csv), [`Week 1/toy_results.csv`](Week%201/toy_results.csv), [`Week 1/results.csv`](Week%201/results.csv).
* **Report:** [`Week 1/Group_5_Component_C_The_Statistical_Layer.pdf`](Week%201/Group_5_Component_C_The_Statistical_Layer.pdf).

---

## 4. Edge-Case, Ambiguity, & Truth-Tie Rules (Preserved Across All Weeks)

Both Week 1 and Week 2 code enforce the following 8 edge-case rules and Truth-Tie specifications:

1. **Rule 1 (Empirical Tie-Breaking):** If two or more classifiers tie for the highest observed mean AUC, select the classifier with the lexicographically smallest `classifier_id` (e.g. `c1` beats `c2`).
2. **Truth-Tie Rule:**
   * **Winner-Correct Rate:** If multiple classifiers share the maximum true AUC in `truth.csv`, any declared winner belonging to the set of true best classifiers receives `is_correct = 1` (no arbitrary penalization).
   * **Paired $t$-Test:** If `true_auc(c1) == true_auc(c2)`, rejections with $p < 0.05$ are classified as **Type I errors (False Positives)** with `power = 0`.
   * **Kendall's Tau:** Evaluated using tie-adjusted Kendall's $\tau_b$.
3. **Rule 2 & 5 (Non-Computable / NaN AUC):** Missing or non-computable fold AUCs invalidate the replicate.
4. **Rule 3 (Fold Integrity):** Every required classifier must have exactly 5 valid folds with integer IDs matching $\{1, 2, 3, 4, 5\}$. Duplicates, non-integers, missing folds, or out-of-range IDs invalidate the replicate.
5. **Rule 4 (Referential Integrity):** Every classifier appearing in results must have a matching numerical `true_auc` in `truth.csv` for that dataset and sample size.
6. **Rule 6 (Degenerate Paired $t$-Test):** Zero variance in differences or mismatched fold IDs yields **NO DETECTION** (`power=0`, `sign_error=0`).
7. **Rule 7 (Strict $p < 0.05$ Threshold):** Detection strictly requires $p < 0.05$; $p \ge 0.05$ is classified as **NO DETECTION**.
8. **Rule 8 (Mean Difference = 0):** Zero mean difference yields **NO DETECTION**.

---

## 5. How to Run & Reproduce

### Run Week 2 Evaluation & 18 Unit Tests:
```bash
python3 "Week 2/component_c.py"
```

### Rebuild Week 2 Deliverable PDF Report:
```bash
python3 "Week 2/generate_pdf_report.py"
```

### Run Week 1 Code:
```bash
python3 "Week 1/component_c.py"
```

---

## 6. Repository Structure

```text
├── README.md                                  <- Main repository documentation (all weeks)
├── .gitignore                                 <- Clean Git ignore rules (Python, macOS, cache)
├── Week 1/
│   ├── component_c.py                         <- Week 1 statistical layer & 16 unit tests
│   ├── generate_synthetic_results.py          <- Week 1 synthetic generator (600 rows)
│   ├── Group_5_Component_C_The_Statistical_Layer.pdf <- Week 1 submitted report PDF
│   ├── results.csv                            <- Week 1 synthetic 5-fold CV results
│   ├── toy_results.csv                        <- Week 1 15-row hand-verification dataset
│   └── truth.csv                              <- Week 1 3-classifier ground truth
└── Week 2/
    ├── component_c.py                         <- Week 2 upgraded engine & 18 unit tests
    ├── generate_pdf_report.py                 <- ReportLab PDF builder
    ├── Group_5_Component_C_Week_2_Report.pdf  <- Week 2 deliverable report PDF
    ├── Group_5_Component_C_Week_2_Report.md   <- Week 2 deliverable markdown report
    ├── four_quantities_summary.csv            <- Week 2 Table 1 machine-readable data
    ├── power_analysis_summary.csv             <- Week 2 Table 2 machine-readable data
    ├── reference_results.csv                  <- Reference experiment dataset (12,000 rows)
    └── reference_truth.csv                    <- Reference ground truth table (30 rows)
```

---

## 7. Dependencies

* Python 3.8+
* `numpy`
* `pandas`
* `scipy`
* `reportlab` (for PDF generation)

Install dependencies via:
```bash
pip install numpy pandas scipy reportlab
```
