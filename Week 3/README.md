# CSIT332 Week 3 — Component C: The Statistical Layer
## Principles of Machine Learning · Semester Project
**Team:** Official Course Group 2 (General Team Identity: Group 5)  
**Members:**
- Akshay
- Jitendra
- Shaun
- Krishang  
**Deliverable:** Week 3 — Component C: Three Protocols, Every Classifier Pair, and the First Sample-Size Calculator  
**Environment:** Python 3.13 / Miniconda (pinned exact versions in `requirements.txt`)

---

### 1. Overview & Assignment Scope

In Week 3, Group 5 delivered **Component C: The Statistical Layer**, evaluating empirical classifier performance across three experimental protocols and establishing the first quantitative sample-size calculator.

All analyses are executed against the official reference datasets:
- [`data/reference_results_v2.csv`](data/reference_results_v2.csv): 81,600 rows (3,600 `split80` + 18,000 `cv5` + 60,000 `repcv5x10`).
- [`data/reference_truth_v2.csv`](data/reference_truth_v2.csv): 42 rows (Truth-at-$n$ for $n \in \{50, 100, 200, 500, 1000, 2000\}$ and Asymptotic Truth).
- [`data/reference_stability_v2.csv`](data/reference_stability_v2.csv): 6 rows (Asymptotic stability metrics).

---

### 2. Module Index

| Module / Asset | Purpose |
| :--- | :--- |
| [`component_c.py`](component_c.py) | Core statistical engine: dual-truth loading, multi-protocol aggregation, tie-breaking, Kendall's $\tau_b$, optimism, and paired statistical tests. |
| [`candidate_analysis.py`](candidate_analysis.py) | **Task 2:** Evaluates all $\binom{6}{k}$ subsets for $k \in \{2, 3, 4, 5, 6\}$ (57 subsets per $n$) on `cv5`, measuring winner-correct vs. $1/k$ and selection optimism growth. |
| [`power_calculator.py`](power_calculator.py) | **Tasks 3 & 4:** Evaluates all 15 classifier pairs across `cv5` (180 conditions) and `repcv5x10` (120 conditions) = 300 conditions. Implements the First Sample-Size Calculator. |
| [`generate_deliverables.py`](generate_deliverables.py) | Master pipeline producing all tabular summaries, publication charts, and markdown deliverables. |
| [`verify_c.py`](verify_c.py) | Automated test suite verifying 16 unit and integration test suites covering all statistical formulas and edge cases (100% pass rate). |
| [`Group_5_Component_C_Week_3_Report.md`](Group_5_Component_C_Week_3_Report.md) | Formal comprehensive deliverable report with discussion, tables, and plot links. |
| [`outputs/`](outputs/) | Directory containing generated tabular deliverables (CSV format). |
| [`plots/`](plots/) | Directory containing high-resolution publication figures (PNG format). |

---

### 3. Key Findings & Scientific Takeaways

1. **Repeated Cross-Validation vs. Genuine Sample Size:**
   - Repeated 5-fold CV (`repcv5x10`) stabilizes ranking concordance over a single split (`split80`), lifting Kendall's $\tau_b$ from $+0.2320$ to $+0.4533$ at $n=50$ and from $+0.5893$ to $+0.8320$ at $n=500$.
   - However, **repeated CV cannot rescue a small sample**. At $n=50$, winner-correct rate remains low ($0.3000$) and selection optimism remains severe ($+0.0842$ vs. $+0.1620$ for `split80`). Resampling a small cohort cannot generate new population draws.

2. **Candidate-Count Selection Optimism:**
   - As candidate count $k$ increases from 2 to 6, selection optimism grows monotonically at every sample size (e.g. from $+0.0648$ at $k=2$ to $+0.1151$ at $k=6$ for $n=50$).
   - Comparing more models draws increasingly from the extreme upper tail of sampling noise, making the nominal winner **less trustworthy**.

3. **Ordinary Paired $t$-Test vs. Nadeau–Bengio Corrected Test:**
   - The ordinary paired $t$-test's independence assumption is problematic on cross-validation folds because overlapping training sets induce positive correlation across fold differences.
   - On 5-fold CV, the ordinary test underestimates standard errors by $33.3\%$ (test statistic inflated by $\sqrt{0.45/0.20} = 1.500\times$ relative to the corrected test), leading to understated uncertainty.
   - The prescribed Nadeau–Bengio correction approximately accounts for this dependence by incorporating the training-to-test ratio ($n_{\text{test}}/n_{\text{train}} = 1/4$). On repeated CV ($J=50$), the ordinary test reports an average power of $67.9\%$, whereas the corrected test indicates an effective power of $27.3\%$, providing a more realistic and conservative evaluation.

4. **The First Calculator:**
   - For a minute true gap of $|\Delta| = 0.01$, power never reaches $80\%$ by $n=2000$ under either test.
   - For a gap of $|\Delta| = 0.02$, power never reaches $80\%$ by $n=2000$ under the Nadeau–Bengio test (peaking at $0.11$ for `cv5` and `repcv5x10`).
   - For a gap of $|\Delta| = 0.05$, $80\%$ power is exceeded at $n=2000$ under `cv5` Ordinary (power $= 0.805$) and exceeded at $n=200$ under `repcv5x10` Ordinary (power $= 0.813$). Under the conservative Nadeau–Bengio test, $80\%$ power is not reached by $n=2000$ under `cv5` (peaking at $0.51$) or by $n=500$ under `repcv5x10` (peaking at $0.44$).

---

### 4. Reproduction Commands

To run the automated verification test suite:
```bash
python "Week 3/verify_c.py"
```

To regenerate all deliverables, plots, and markdown reports:
```bash
python "Week 3/generate_deliverables.py"
```
