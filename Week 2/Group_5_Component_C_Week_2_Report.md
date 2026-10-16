# Component C (Group 5) — Week 2 Report
## CSIT332: Principles of Machine Learning · Semester Project
**Author:** Group 5 (Component C: The Statistical Evaluation Layer)  
**Deliverable:** Week 2 — Benchmark Evaluation on Reference Data & The Dual Nature of Truth  
**Target Code & Data:** `component_c.py`, `reference_results.csv`, `reference_truth.csv`

---

## 1. Executive Summary

During Week 1 pilot evaluations on the reference machinery, applying the preliminary benchmark definitions yielded a **negative optimism value of −0.035** at the smallest sample size ($N=50$). By definition, optimism is intended to measure inflation—the amount by which an empirical winning score overstates reality. A negative value cannot occur if the metric is measuring what we think it measures (pure selection inflation). Rather than being mathematically impossible as a simple difference, this negative result revealed a conceptual category error: Version B combined genuine selection bias with the finite-sample training deficit, misinterpreting the resulting composite quantity as pure selection optimism.

Investigation revealed that two opposing phenomena were conflated into a single metric:
1. **Model Selection Bias (Optimism):** Declaring the empirical maximum over multiple noisy estimators capitalizes on chance, pushing the winning score *upward*.
2. **Finite-Sample Skill Deficit:** A classifier fitted on small training partitions (e.g., 40 rows in 5-fold CV) achieves lower true generalization performance than the same model trained on abundant data (12,600 rows), pushing its score *downward*.

In Week 1, the winner's score was evaluated against the **asymptotic benchmark** ($N=12,600$). On 50 rows, the finite-sample deficit swamped the selection bias ($+0.106 - 0.141 = -0.035$). 

**The Week 2 Correction:** Ground truth is now measured at **each sample size** ($n \in \{50, 100, 200, 500\}$) by evaluating models trained on the exact cross-validation training folds against a permanent frozen test set of 6,000 independent rows (**Version A**), alongside the abundant-data benchmark (**Version B**). 

This report provides the full evaluation of the 12,000-row reference dataset, presents the comparative tables for both versions, analyzes paired $t$-test power across close and wide classifier gaps, and explains the theoretical foundations in plain English.

---

## 2. Why the Two Versions of Truth Give Different Answers (Plain English Explanation)

### 2.1 The Practical Question of Finite-Sample Research
Consider a biomedical researcher who collects 50 patient records and runs 5-fold cross-validation to select the most promising diagnostic algorithm. Her core question is:
> *Does her empirical comparison support declaring a winner, and by how much does her winning score overstate reality?*

Her sample size of 50 is an unalterable constraint of her clinical reality. She is **not** asking which model would win if she had 12,600 patients, because she will never possess 12,600 patients. If a simpler algorithm genuinely outperforms a complex model at 50 patients, selecting that simpler algorithm is the **objectively correct scientific decision**.

### 2.2 Version A: Isolating Pure Selection Bias
In **Version A**, ground truth is established at her exact sample size ($n$). Each classifier is fitted on the four-fifths of the sample seen during cross-validation (40 rows) and tested on the frozen 6,000 held-out rows. The winner's cross-validated score is then compared against that same model's honest generalization score at $n$.

Because cross-validation estimates are noisy random variables, taking the maximum over six candidate models capitalizes on positive sampling error:
* At $n=50$, mean optimism is **$+0.1059$** ($+10.6\%$ AUC inflation).
* As sample size grows to $n=100$, $200$, and $500$, cross-validation variance shrinks, and optimism monotonically contracts to **$+0.0559$**, **$+0.0406$**, and **$+0.0220$**.
* The **winner-correct rate** generally improves as sample size increases, rising from **$43.0\%$** at $n = 50$ to **$69.0\%$** at $n = 500$. However, it dips slightly to **$41.0\%$** at $n = 100$ before increasing at the larger sample sizes ($47.0\%$ at $n=200$ and $69.0\%$ at $n=500$), as larger samples provide a clearer signal to identify the genuinely best model.

Version A behaves exactly as statistical theory dictates: optimism is **strictly positive and shrinks toward zero**, isolating the researcher's model-selection uncertainty.

### 2.3 Version B: Conflating Two Separate Phenomena
In **Version B**, the declared winner's cross-validated score is compared against its abundant-data performance ($N=12,600$). Mathematically:

$$\text{Observed}(n) - \text{True}(\infty) = \underbrace{[\text{Observed}(n) - \text{True}(n)]}_{\text{Selection Bias (Optimism } > 0\text{)}} + \underbrace{[\text{True}(n) - \text{True}(\infty)]}_{\text{Finite-Sample Skill Deficit } (< 0)}$$

* The first term is selection-induced optimism ($+0.1059$ at $n=50$), which pushes the score **up**.
* The second term is the skill deficit ($0.6536 - 0.7947 = -0.1411$ at $n=50$), which pulls the score **down** because a model trained on 40 rows lacks the training data to reach its full asymptotic capability.
* Adding them together yields:
  $$+0.1059 + (-0.1411) = -0.0352$$

The finite-sample deficit swamped the selection bias. Version B reports this combined artifact, creating the illusion of negative optimism.

### 2.4 The Inductive Bias Tradeoff & Penalizing Right Decisions
Different classifiers have fundamentally different learning curves:
* **Logistic Regression (`logreg`)** has strong inductive bias and low variance. It reaches near-peak skill with very few samples: true AUC = **0.690** at $n=50$, rising modestly to **0.789** asymptotically.
* **Gradient Boosting (`gb`)** has high capacity and high variance. At $n=50$, it severely overfits, achieving only **0.636** true AUC; only with abundant data ($N=12,600$) does it reach **0.837**.

At 50 samples, logistic regression beats gradient boosting by **$+0.0544$** in true generalization. When the researcher picks logistic regression, she has made the right choice. 
* **Version A** credits her decision (**$43\%$ winner-correct rate**).
* **Version B** insists that gradient boosting is the "truth" simply because it wins with 12,600 rows. It marks her correct decision as an error, causing the reported winner-correct rate to collapse to **$15\%$** at $n=50$ and **$8\%$** at $n=500$.

Version B penalizes the researcher for a choice that was scientifically sound, answering an irrelevant counterfactual rather than measuring the finite-sample uncertainty she actually faces.

---

## 3. Results Table: Four Quantities × Four Sample Sizes

Table 1 reports the four core quantities computed across the 100 replicates (12,000 rows) of `reference_results.csv` against `reference_truth.csv`.

### Table 1: Experimental Summary Across All Sample Sizes

| Sample Size ($n$) | Winner-Correct (Ver A) | Winner-Correct (Ver B) | Mean Kendall $\tau_b$ (Ver A) | Mean Kendall $\tau_b$ (Ver B)* | Mean Optimism (Ver A) | Mean Optimism (Ver B) | Genuinely Best Model at $n$ [True AUC] | Empirical Winner Distribution (100 Replicates) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n = 50$** | **0.4300** | 0.1500 | **+0.3166** | +0.0177 | **+0.1059** | −0.0352 | `logreg` [0.6903] | `logreg` 43%, `gb` 15%, `svm` 13%, `rf` 13%, `knn` 12%, `dtree` 4% |
| **$n = 100$** | **0.4100** | 0.1200 | **+0.5091** | +0.2008 | **+0.0559** | −0.0439 | `logreg` [0.7232] | `logreg` 41%, `svm` 32%, `gb` 12%, `rf` 10%, `knn` 5% |
| **$n = 200$** | **0.4700** | 0.0600 | **+0.6908** | +0.3358 | **+0.0406** | −0.0216 | `svm` [0.7544] | `svm` 47%, `logreg` 32%, `rf` 13%, `gb` 6%, `dtree` 1%, `knn` 1% |
| **$n = 500$** | **0.6900** | 0.0800 | **+0.8063** | +0.4728 | **+0.0220** | −0.0091 | `svm` [0.7856] | `svm` 69%, `logreg` 23%, `gb` 8% |

*\*Note on Kendall's $\tau_b$: Version A (against true AUC at $n$) is the primary finite-sample metric requested by the assignment. Version B (against asymptotic AUC) is included as supplementary context to illustrate the divergence in ordinal rankings at small sample sizes. In Version B, the asymptotic target winner is `gb` with true AUC = 0.836803.*

### Key Observations:
1. **Optimism Behavior:** Version A optimism is strictly positive and monotonic: $+0.1059 \to +0.0559 \to +0.0406 \to +0.0220$. Version B optimism remains negative across all sample sizes ($-0.0352$ at $n=50$).
2. **Winner-Correct Rate:** The winner-correct rate generally improves as sample size increases, rising from 43% at $n = 50$ to 69% at $n = 500$. However, it dips slightly to 41% at $n = 100$ before increasing at the larger sample sizes (47% at $n=200$, 69% at $n=500$). Under Version B, winner-correct stays suppressed ($6\% - 15\%$) because the asymptotic winner (`gb`) is suboptimal on small data.
3. **Rank Correlation (Kendall's $\tau_b$):** Version A rank correlation climbs steadily from $+0.3166$ to $+0.8063$, confirming that cross-validation rankings converge toward the true model ranking at that sample size. Version B correlation is nearly zero ($+0.0177$) at $n=50$, demonstrating that small-sample rankings share almost no ordinal structure with abundant-data rankings.
4. **Winner Distribution Consolidation:** At $n=50$, sampling variance allows every single classifier to win in at least 4% of replicates. By $n=500$, the field consolidates: `svm` wins 69% and `logreg` wins 23%, perfectly matching the top two true models at $n=500$.

---

## 4. Task 3: Paired $t$-Test Power Analysis (Close Pair vs. Wide Pair)

To examine why sample size ($n$) and effect size ($\Delta$) must be considered jointly, we evaluate 5-fold paired $t$-tests at $\alpha = 0.05$ across all 100 replicates for:
* **Close Pair:** Differing by less than 0.01 in true AUC (`gb` vs `knn`, true $\Delta = +0.0016$ at $n=50$).
* **Wide Pair:** Separated by more than 0.05 in true AUC (`logreg` vs `dtree`, true $\Delta > 0.12$ across all $n$).

Table 2 reports Power and Sign-Error rates under Version A (truth at sample size $n$), alongside Version B (asymptotic truth) as complete comparative context.

### Table 2: Paired $t$-Test Power and Sign-Error Rate

| Classifier Pair | Sample Size ($n$) | True Gap at $n$ ($\Delta$ AUC) | Observed Mean $\Delta$ | Power (Ver A) [Correct Detection] | Sign-Error (Ver A) [Wrong Direction] | Power (Ver B)* [Asymptotic Ref] | Sign-Error (Ver B)* [Asymptotic Ref] | Methodological Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`gb` vs `knn`**<br/>*(Close Pair: $\Delta < 0.01$ at $n=50$)* | 50<br/>100<br/>200<br/>500 | +0.0016<br/>+0.0183<br/>+0.0406<br/>+0.0634 | +0.0034<br/>+0.0263<br/>+0.0383<br/>+0.0667 | 0.0500 (5%)<br/>0.0800 (8%)<br/>0.1400 (14%)<br/>0.3800 (38%) | **0.0400 (4%)**<br/>0.0100 (1%)<br/>0.0000 (0%)<br/>0.0000 (0%) | 0.0500<br/>0.0800<br/>0.1400<br/>0.3800 | 0.0400<br/>0.0100<br/>0.0000<br/>0.0000 | At $n=50$, true difference is negligible. Statistical power is at nominal $\alpha$ (5%), while sign error is 4%. A significant test is essentially a coin flip. |
| **`logreg` vs `dtree`**<br/>*(Wide Pair: $\Delta > 0.12$ across all $n$)* | 50<br/>100<br/>200<br/>500 | +0.1290<br/>+0.1494<br/>+0.1558<br/>+0.1415 | +0.1342<br/>+0.1504<br/>+0.1682<br/>+0.1469 | 0.1800 (18%)<br/>0.4200 (42%)<br/>0.5200 (52%)<br/>**0.8700 (87%)** | 0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%) | 0.1800<br/>0.4200<br/>0.5200<br/>0.8700 | 0.0000<br/>0.0000<br/>0.0000<br/>0.0000 | Large gap completely prevents sign errors (0%). Power scales rapidly from 18% to 87%, achieving standard clinical power ($\ge 80\%$) at $n=500$. |

*\*Note: Version B power and sign-error rates evaluate detection against the asymptotic truth direction (where gb > knn by +0.0992, and logreg > dtree by +0.0507). Both versions agree on direction for these two pairs.*

### Supplementary Analysis: The Lead Reversal Pair (`svm` vs `logreg`)
To illustrate how truth changes across sample sizes, we also analyzed `svm` vs `logreg`, where `logreg` is superior at small samples ($n=50, 100$), but `svm` takes the lead at $n=200$ and $n=500$:

| Pair | Sample Size ($n$) | True Gap at $n$ | Observed Mean $\Delta$ | Power (Ver A) | Sign-Error (Ver A) | Power (Ver B) | Sign-Error (Ver B) | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`svm` vs `logreg`** | 50 | −0.0981 | −0.1055 | 0.0300 (3%) | 0.0100 (1%) | **0.0100 (1%)** | **0.0300 (3%)** | `logreg` is truly superior at $n=50$. Ver A credits favoring `logreg`. Ver B penalizes it! |
| | 100 | −0.0190 | −0.0214 | 0.0500 (5%) | 0.0400 (4%) | 0.0400 (4%) | 0.0500 (5%) | `logreg` leads by 0.019; high noise regime. |
| | 200 | +0.0021 | +0.0074 | 0.0300 (3%) | 0.0100 (1%) | 0.0300 (3%) | 0.0100 (1%) | Near-tie ($\Delta = 0.0021$); `svm` takes slight lead. |
| | 500 | +0.0121 | +0.0186 | 0.1300 (13%) | 0.0100 (1%) | 0.1300 (13%) | 0.0100 (1%) | `svm` solidifies lead. |

### Deep Dive: Statistical Mechanics of the Contrast
1. **The Risk of Sign Errors in Small Samples:** When two classifiers are separated by a near-tie ($\Delta = 0.0016$ for `gb` vs `knn` at $n=50$), sampling variance easily overwhelms the true difference. Out of 9 statistically significant rejections ($p < 0.05$), 5 favored `gb` (power = 5%) and 4 favored `knn` (sign error = 4%). A researcher declaring a winner based on $p < 0.05$ has a **$44.4\%$ probability of naming the inferior model**.
2. **Wide Gaps Eliminate Sign Errors:** For `logreg` vs `dtree` ($\Delta \approx 0.14$), not a single replicate out of 400 produced a sign error. Even at $n=50$, when power was limited to 18%, every single significant rejection correctly favored `logreg`.
3. **Power Scaling:** Power depends on the non-centrality parameter $\lambda = \frac{\Delta \sqrt{K}}{\sigma_D}$. When $\Delta$ is minute, increasing $N$ from 50 to 100 provides negligible power gains ($5\% \to 8\%$). When $\Delta$ is large, increasing $N$ drives standard error down rapidly, propelling power from 18% to 87%.

---

## 5. Software Architecture & Verification

The updated Python implementation in [`component_c.py`](component_c.py) maintains full backward compatibility with Week 1 edge-case rules while introducing dual-truth evaluation:

* **Modular Truth Slicing:** Dynamically handles the new 4-column `truth.csv` schema (`dataset_id`, `classifier_id`, `n_sub`, `true_auc`), filtering for sample-size truth (`mode='per_sample'`) and asymptotic truth (`mode='asymptotic'`).
* **Preservation of Edge-Case Rules:**
  * **Rule 1:** Lexicographically smallest classifier ID breaks exact empirical ties.
  * **Truth-Tie Rules:** Preserves tied ground-truth winners without arbitrary penalization; handles truth-ties in paired $t$-tests as Type I errors rather than power.
  * **Rules 2, 3, 5:** Enforces exact fold IDs $\{1, 2, 3, 4, 5\}$, integer fold types, and complete rejection of missing/NaN fold scores.
  * **Rule 4:** Full referential integrity between results and ground truth.
  * **Rules 6, 7, 8:** Strict $p < 0.05$ threshold (specifically verified at boundary $p \ge 0.05$), zero-variance difference protection, and mean-difference = 0 rejection.
* **18 Automated Unit Tests:** Passed 100% of test suites covering all edge cases, theoretical properties, and dual-version outputs.

---

## 6. How to Run & Reproduce

All files are self-contained in the `Week 2` directory.

### 1. Run Statistical Evaluation & Automated Test Suite
```bash
python3 "Week 2/component_c.py"
```

### 2. Generate the PDF Deliverable
```bash
python3 "Week 2/generate_pdf_report.py"
```

### 3. Generated Deliverables in `Week 2/`:
* `component_c.py`: Production-grade statistical engine with 18 automated unit tests.
* `Group_5_Component_C_Week_2_Report.pdf`: Publication-quality formatted PDF deliverable.
* `Group_5_Component_C_Week_2_Report.md`: Full Markdown documentation and analysis.
* `four_quantities_summary.csv`: Machine-readable results for Table 1.
* `power_analysis_summary.csv`: Machine-readable results for Table 2.
* `reference_results.csv`: 12,000-row reference results file.
* `reference_truth.csv`: 30-row ground-truth benchmark file.
