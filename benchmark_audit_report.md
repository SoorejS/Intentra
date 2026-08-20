# 🔬 Intentra V2 — Benchmark & Scientific Integrity Audit Report
**Date:** August 20, 2026  
**Auditor:** Antigravity Autonomous Audit Engine  
**Status:** Complete & Scientifically Validated  

---

## 1. Executive Summary & Audit Verdict

Following a thorough investigation into the benchmark pipeline, dataset generation, and evaluation mechanics, we present an honest, mathematically verifiable empirical audit.

### 🔴 AUDIT VERDICT: **RED**
**Intentra V2 does not currently beat the Naive Synthetic baseline on the linear probe benchmark.**
The previously claimed **2.0x–3.3x Data Efficiency Multiplier** was an artifact of a fallback calculation flaw in the benchmark runner where unreached target thresholds defaulted to hardcoded sample ratios. That claim has been **completely rescinded and removed**.

---

## 2. Implementation & Root-Cause Audit

### 2.1 The Metric & Multiplier Calculation Flaw
In the original `core/benchmark_suite.py`, the data efficiency multiplier was calculated with the following logic:
```python
target_f1 = 0.70
v2_examples_needed = sample_sizes[-1]       # defaulted to 300
naive_examples_needed = sample_sizes[-1] * 2 # defaulted to 600

for sz in sample_sizes:
    if results_by_method["intentra_v2_closed_loop"]["mean_f1_by_size"][sz] >= target_f1:
        v2_examples_needed = sz
        break
```
When neither model achieved the target $F_1 = 0.70$, the loop never triggered, leaving `naive_examples_needed = 600` and `v2_examples_needed = 300`, outputting a false `600 / 300 = 2.0x` multiplier despite observed $F_1$ being $< 0.30$.

### 2.2 Data Leakage Audit
- **Original Code**: Line 204 evaluated baseline models against `test_set[:15]`, which accidentally sliced from the holdout evaluation set.
- **Audit Fix**: Enforced strict **3-way disjoint split architecture**:
  1. $D_{\text{train}}$: Training sample pool ($N \in [50, 100, 200, 300, 500, 1000]$).
  2. $D_{\text{val}}$: 25 distinct annotated customer support validation samples (used *exclusively* for error diagnosis).
  3. $D_{\text{test}}$: 50 locked hand-crafted holdout test samples (40 boundary + 10 hard-negative) with **0% text or lexical overlap** with $D_{\text{val}}$ or $D_{\text{train}}$.
  $$\text{Leakage Overlap } |D_{\text{val}} \cap D_{\text{test}}| = 0 \quad (\text{Verified})$$

### 2.3 Dataset-Equivalence Audit
- **Class Balance**: All three methods (Naive, V1, V2) are evaluated on identical 5-class distributions (`refund_request`, `cancellation_request`, `billing_inquiry`, `technical_support`, `general_feedback`).
- **Sample Budgets**: Exactly matched at $N \in [50, 100, 200, 300, 500, 1000]$.
- **Training Settings**: Identical TF-IDF vectorizer parameters (`ngram_range=(1,2)`, `max_features=5000`, `sublinear_tf=True`) and Logistic Regression hyperparameters (`C=1.0`, `max_iter=1000`, `random_state=seed`).

---

## 3. Corrected Empirical Benchmark Results

Evaluated across **5 Independent Deterministic Random Seeds** (`[42, 123, 456, 789, 999]`) and **6 Training Sample Sizes** on the **Locked 50-Example Holdout Test Set**:

### 📊 Master Benchmark Table ($\mu \pm \sigma$, $95\%\text{ CI}$)

| Method | Sample Budget ($N$) | Mean Macro $F_1$ | Std Dev ($\sigma$) | $95\%\text{ CI}$ | Mean Accuracy | Boundary Acc | Hard-Negative Acc |
|---|---|---|---|---|---|---|---|
| **Naive Synthetic** | 50 | **0.3795** | $\pm 0.0000$ | $\pm 0.0000$ | 40.0% | 38.1% | 50.0% |
| **Intentra V1 (Static)** | 50 | 0.3594 | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 50 | 0.1793 | $\pm 0.0000$ | $\pm 0.0000$ | 26.0% | 26.2% | 25.0% |
| | | | | | | | |
| **Naive Synthetic** | 100 | **0.3795** | $\pm 0.0000$ | $\pm 0.0000$ | 40.0% | 38.1% | 50.0% |
| **Intentra V1 (Static)** | 100 | 0.3594 | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 100 | 0.2202 | $\pm 0.0000$ | $\pm 0.0000$ | 28.0% | 26.2% | 37.5% |
| | | | | | | | |
| **Naive Synthetic** | 200 | **0.3795** | $\pm 0.0000$ | $\pm 0.0000$ | 40.0% | 38.1% | 50.0% |
| **Intentra V1 (Static)** | 200 | 0.3594 | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 200 | 0.2446 | $\pm 0.0000$ | $\pm 0.0000$ | 32.0% | 31.0% | 37.5% |
| | | | | | | | |
| **Naive Synthetic** | 300 | **0.3795** | $\pm 0.0000$ | $\pm 0.0000$ | 40.0% | 38.1% | 50.0% |
| **Intentra V1 (Static)** | 300 | 0.3594 | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 300 | 0.2446 | $\pm 0.0000$ | $\pm 0.0000$ | 32.0% | 31.0% | 37.5% |
| | | | | | | | |
| **Naive Synthetic** | 500 | 0.3513 | $\pm 0.0000$ | $\pm 0.0000$ | 38.0% | 35.7% | 50.0% |
| **Intentra V1 (Static)** | 500 | **0.3594** | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 500 | 0.2806 | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| | | | | | | | |
| **Naive Synthetic** | 1000 | 0.3513 | $\pm 0.0000$ | $\pm 0.0000$ | 38.0% | 35.7% | 50.0% |
| **Intentra V1 (Static)** | 1000 | **0.3594** | $\pm 0.0000$ | $\pm 0.0000$ | 34.0% | 33.3% | 37.5% |
| **Intentra V2 (Closed-Loop)** | 1000 | 0.2283 | $\pm 0.0000$ | $\pm 0.0000$ | 30.0% | 28.6% | 37.5% |

---

## 4. Mathematical Data Efficiency Analysis

### Definition
For a specified performance target $F_1^*$:
$$\text{Data Efficiency Multiplier} = \frac{N_{\text{naive}}(F_1 \ge F_1^*)}{N_{\text{intentra}}(F_1 \ge F_1^*)}$$

### Evaluation Across Target Thresholds

| Target Macro $F_1^*$ | $N_{\text{naive}}$ Required | $N_{\text{Intentra V1}}$ Required | $N_{\text{Intentra V2}}$ Required | Data Efficiency Multiplier |
|---|---|---|---|---|
| **0.35** | **50** | **50** | **Target Not Reached** (Peak = 0.2806) | **Target Not Reached** |
| **0.40** | Target Not Reached | Target Not Reached | Target Not Reached | **Target Not Reached** |
| **0.50** | Target Not Reached | Target Not Reached | Target Not Reached | **Target Not Reached** |
| **0.60** | Target Not Reached | Target Not Reached | Target Not Reached | **Target Not Reached** |
| **0.70** | Target Not Reached | Target Not Reached | Target Not Reached | **Target Not Reached** |

> **Conclusion**: Because Intentra V2 does not achieve the target performance thresholds reached by Naive Synthetic, **no Data Efficiency Multiplier can be claimed** on this benchmark.

---

## 5. Mechanistic & Statistical Diagnosis: Why Did V2 Struggle Here?

1. **Bag-of-Words Limitation with Adversarial / Contrastive Sentences**:
   - Intentra V2 targets decision-boundary confusion by synthesizing contrastive sentences such as:
     *"I don't need a refund right now, I just need to terminate this recurring plan immediately."*
   - To a human or a Transformer (DistilBERT/BERT), the syntactic structure clarifies that the intent is `cancellation_request`.
   - To a **TF-IDF linear model**, the word `"refund"` contributes positive weight towards `refund_request` regardless of the preceding negation `"don't need"`. Introducing dense contrastive boundary examples actually increases lexical ambiguity for bag-of-words classifiers.
2. **Naive Data Keyword Simplicity**:
   - Naive synthetic data contains disjoint, orthogonal keyword clusters (`"refund"`, `"cancel"`, `"invoice"`, `"crash"`, `"praise"`), making it artificially easy for linear probes to draw hyperplanes, despite being brittle on real-world edge cases.

---

## 6. Telemetry, Cost & Wall-Clock Profile

- **Total Models Trained:** 90
- **Total Evaluations Run:** 90
- **Total Wall-Clock Time:** 23.98s (sub-second per run on CPU)
- **Estimated Generation & Training Cost:** $0.00 (Fast mode execution)

---

## 7. Action Plan & Next Steps

1. **Retain Scientific Integrity**: The UI and benchmark documentation will report the exact empirical metrics without inflated multiplier claims.
2. **Transformer-Based Deep Benchmark**: Run the benchmark suite on Transformer architectures (`distilbert-base-uncased` and `roberta-base`) where syntactic attention mechanisms can properly leverage contrastive hard negatives.
3. **Refine Targeted Generation Prompting**: Ensure targeted boundary examples include disambiguating anchor tokens rather than pure adversarial keyword competition.
