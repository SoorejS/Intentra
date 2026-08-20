# Intentra V2 — Transformer (DistilBERT) Benchmark & Mechanistic Audit Report

**Date**: August 20, 2026  
**Auditor**: Antigravity Automated Scientific Audit Engine  
**Target Architecture**: Intentra V2 Closed-Loop Optimization Engine  
**Downstream Evaluators**:
1. **Linear Probe**: TF-IDF (1000 max features) + Logistic Regression ($C=1.0$)
2. **Contextual Transformer**: DistilBERT (`distilbert-base-uncased`, Sequence Classification)

---

## 1. Executive Verdict

> [!CAUTION]
> ### Empirical Verdict: RED / ADVANTAGE NOT DEMONSTRATED
> Under strictly controlled, zero-leakage benchmark conditions across 5 random seeds (`42, 123, 456, 789, 999`) and 6 training budgets ($N \in \{50, 100, 200, 300, 500, 1000\}$), **Intentra V2 did not demonstrate an advantage over Naive Synthetic data on DistilBERT**, just as it did not on the TF-IDF probe.
> 
> * **DistilBERT Peak Macro $F_1$ ($N=1000$)**:
>   * **Naive Synthetic**: $\mathbf{0.7882 \pm 0.0111}$ ($\text{CI}_{95} = 0.0098$)
>   * **Intentra V1**: $0.5303 \pm 0.0135$ ($\text{CI}_{95} = 0.0118$)
>   * **Intentra V2**: $0.6030 \pm 0.0407$ ($\text{CI}_{95} = 0.0357$)
> * **Data-Efficiency Multiplier**: **None / Not Reached**. Naive Synthetic requires far fewer examples to reach every target performance threshold ($F_1 \ge 0.40, 0.50, 0.60, 0.70, 0.75$).
> * **Hypothesis Verdict**: **REFUTED FOR SUPERIORITY**. While switching from TF-IDF to DistilBERT allowed Intentra V2 to scale significantly better than Intentra V1 at $N \ge 500$ ($0.6030$ vs $0.5303$), **Naive Synthetic remained strictly superior at every budget**.

---

## 2. Experimental Methodology & Control Setup

To eliminate all experimental confounding variables, the transformer benchmark applied identical experimental controls:

1. **Exact 3-Way Disjoint Split**:
   * **$D_{\text{train}}$**: Sized to budget $N \in \{50, 100, 200, 300, 500, 1000\}$.
   * **$D_{\text{val}}$**: 25 dedicated diagnostic examples used exclusively by Intentra V2's error analyzer.
   * **$D_{\text{test}}$**: 50 locked, unseen holdout test examples (10 per class) with dedicated slice tags (21 boundary cases, 8 hard negatives, 21 canonical cases).
2. **Leakage Verification**:
   * $|D_{\text{val}} \cap D_{\text{test}}| = 0$ (verified: 0 overlapping strings).
   * $|D_{\text{train}} \cap D_{\text{test}}| = 0$ (verified: 0 overlapping strings).
3. **Random Seeds**: Exactly 5 seeds (`[42, 123, 456, 789, 999]`) per cell, yielding $6 \times 3 \times 5 = 90$ total DistilBERT fine-tuning and evaluation runs.
4. **Model Architecture**:
   * Backbone: `distilbert-base-uncased`.
   * Head: PyTorch `AutoModelForSequenceClassification` with 5 class logits.
   * Learning Rate: $5 \times 10^{-5}$, Weight Decay: $0.01$, Epochs: 3, Optimizer: AdamW.
   * Total Wall-Clock Execution Time: **6,886.85 seconds** (114.78 minutes).

---

## 3. Empirical Results: TF-IDF vs. DistilBERT Side-by-Side

### Full Comparative Table (Mean Macro $F_1 \pm \text{Std}$)

| Budget ($N$) | Method | TF-IDF + Logistic Regression | DistilBERT (Transformer) | $\Delta$ (DistilBERT vs TF-IDF) |
|---|---|:---:|:---:|:---:|
| **50** | **Naive Synthetic** | $\mathbf{0.3795 \pm 0.000}$ | $\mathbf{0.3311 \pm 0.0918}$ | $-0.0484$ |
| | **Intentra V1** | $0.3594 \pm 0.000$ | $0.3280 \pm 0.0854$ | $-0.0314$ |
| | **Intentra V2** | $0.1793 \pm 0.000$ | $0.0667 \pm 0.0000$ | $-0.1126$ |
| **100** | **Naive Synthetic** | $\mathbf{0.3795 \pm 0.000}$ | $\mathbf{0.4925 \pm 0.1749}$ | $+0.1130$ |
| | **Intentra V1** | $0.3594 \pm 0.000$ | $0.3620 \pm 0.1136$ | $+0.0026$ |
| | **Intentra V2** | $0.2202 \pm 0.000$ | $0.1297 \pm 0.0441$ | $-0.0905$ |
| **200** | **Naive Synthetic** | $\mathbf{0.3795 \pm 0.000}$ | $\mathbf{0.6730 \pm 0.0484}$ | $+0.2935$ |
| | **Intentra V1** | $0.3594 \pm 0.000$ | $0.4357 \pm 0.0675$ | $+0.0763$ |
| | **Intentra V2** | $0.2446 \pm 0.000$ | $0.1470 \pm 0.0207$ | $-0.0976$ |
| **300** | **Naive Synthetic** | $\mathbf{0.3795 \pm 0.000}$ | $\mathbf{0.7236 \pm 0.0916}$ | $+0.3441$ |
| | **Intentra V1** | $0.3594 \pm 0.000$ | $0.4275 \pm 0.0567$ | $+0.0681$ |
| | **Intentra V2** | $0.2446 \pm 0.000$ | $0.2428 \pm 0.0605$ | $-0.0018$ |
| **500** | **Naive Synthetic** | $\mathbf{0.3513 \pm 0.000}$ | $\mathbf{0.7807 \pm 0.0304}$ | $\mathbf{+0.4294}$ |
| | **Intentra V1** | $0.3594 \pm 0.000$ | $0.5178 \pm 0.0322$ | $+0.1584$ |
| | **Intentra V2** | $0.2806 \pm 0.000$ | $0.5089 \pm 0.0645$ | $+0.2283$ |
| **1000** | **Naive Synthetic** | $0.3513 \pm 0.000$ | $\mathbf{0.7882 \pm 0.0111}$ | $\mathbf{+0.4369}$ |
| | **Intentra V1** | $\mathbf{0.3594 \pm 0.000}$ | $0.5303 \pm 0.0135$ | $+0.1709$ |
| | **Intentra V2** | $0.2283 \pm 0.000$ | $0.6030 \pm 0.0407$ | $+0.3747$ |

---

## 4. Slice Performance Breakdown (DistilBERT)

### Boundary Accuracy vs. Hard-Negative Accuracy

| Budget ($N$) | Method | Overall Accuracy | Boundary Slice Acc ($N=21$) | Hard-Negative Slice Acc ($N=8$) |
|---|---|:---:|:---:|:---:|
| **50** | Naive Synthetic | **40.0%** | **39.5%** | 42.5% |
| | Intentra V1 | 39.2% | 36.7% | **52.5%** |
| | Intentra V2 | 20.0% | 21.4% | 12.5% |
| **100** | Naive Synthetic | **56.0%** | **55.2%** | **60.0%** |
| | Intentra V1 | 45.2% | 43.8% | 52.5% |
| | Intentra V2 | 24.4% | 24.3% | 25.0% |
| **200** | Naive Synthetic | **70.4%** | **70.0%** | **72.5%** |
| | Intentra V1 | 48.8% | 47.6% | 55.0% |
| | Intentra V2 | 25.6% | 24.8% | 30.0% |
| **300** | Naive Synthetic | **74.0%** | **74.8%** | **70.0%** |
| | Intentra V1 | 48.8% | 47.6% | 55.0% |
| | Intentra V2 | 33.6% | 34.3% | 30.0% |
| **500** | Naive Synthetic | **78.8%** | **79.5%** | **75.0%** |
| | Intentra V1 | 55.2% | 53.8% | 62.5% |
| | Intentra V2 | 53.6% | 56.7% | 37.5% |
| **1000** | Naive Synthetic | **79.2%** | **80.0%** | **75.0%** |
| | Intentra V1 | 55.6% | 54.8% | 60.0% |
| | Intentra V2 | 61.6% | 63.3% | 52.5% |

---

## 5. Hypothesis Testing: Semantic Classifier Capacity

### The Stated Hypothesis
> *"Intentra V2's targeted boundary/contrastive generation is more useful for semantically capable classifiers than for lexical linear classifiers."*

### Empirical Analysis of Hypothesis

1. **Relative Scaling (Supported)**:
   * On TF-IDF, V2 reached a maximum of $0.2806$ and decayed to $0.2283$ at $N=1000$.
   * On DistilBERT, V2 scaled rapidly with sample size: $0.0667 \to 0.1297 \to 0.1470 \to 0.2428 \to 0.5089 \to 0.6030$.
   * At $N=1000$, Intentra V2 ($0.6030$) clearly outperformed Intentra V1 ($0.5303$).
2. **Absolute Superiority (Refuted)**:
   * Despite scaling better than V1, Intentra V2 lagged behind Naive Synthetic at **every single sample budget** ($N=50$ through $N=1000$).
   * Naive Synthetic on DistilBERT reached $\mathbf{0.7882}$ Macro $F_1$, beating Intentra V2 by $+0.1852$ ($+30.7\%$ relative advantage).
   * **Conclusion**: The hypothesis that "Intentra V2 will beat Naive Synthetic once evaluated on a transformer" is **empirically disproven under these experimental conditions**.

---

## 6. Generator Quality Audit & Root Cause Analysis

Why did Intentra V2 underperform Naive Synthetic on both TF-IDF and DistilBERT?

### Direct Inspection of Generated Training Data

Inspection of `scratch/distilbert_raw_data.json` reveals the exact mechanism:

```json
[
  {
    "text": "Does initiating a cancellation_request automatically prevent any future billing_inquiry? (Case #23)",
    "label": "cancellation_request",
    "type": "boundary"
  },
  {
    "text": "Although I previously inquired about billing_inquiry, I want to confirm my request for cancellation_request. (Case #24)",
    "label": "cancellation_request",
    "type": "hard_negative"
  },
  {
    "text": "I don't need a billing_inquiry anymore; just go ahead and finalize the cancellation_request. (Case #25)",
    "label": "cancellation_request",
    "type": "hard_negative"
  }
]
```

### Key Failure Modes Discovered:

1. **Dual-Keyword Contamination in Low-Data Regimes**:
   * In programmatic/synthetic boundary generation, contrastive sentences frequently contain keywords for **two competing classes simultaneously** (e.g., both `"billing_inquiry"` and `"cancellation_request"` in the same sentence).
   * When $N=50$ or $N=100$, over $40\%$ of the training dataset consists of sentences mentioning both classes.
   * For both linear probes and early transformer epochs, this produces high gradient variance and label confusion, leading to classifier collapse (predicting the same class for all test cases, yielding $F_1 \approx 0.0667$).
2. **Naive Synthetic Purity Advantage**:
   * Naive synthetic data contains simple, clean, unambiguous sentences (e.g., *"I want to cancel my subscription"* or *"Why was my credit card charged twice?"*).
   * Even though naive data lacks subtle boundary cases, it gives the classifier an extremely clean, stable semantic anchor for each class centroid. DistilBERT quickly maps these to separated clusters in embedding space ($F_1 = 0.7882$).
3. **The "Curse of Premature Hardening"**:
   * Introducing adversarial boundary cases and hard negatives **before** the classifier has established strong canonical class representations degrades the representation geometry rather than refining it.

---

## 7. Data Efficiency Multiplier Re-Verification

| Target Macro $F_1$ | $N_{\text{Naive}}$ Required | $N_{\text{V1}}$ Required | $N_{\text{V2}}$ Required | True Data-Efficiency Multiplier |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 0.40$** | **100** | 200 | 500 | **$< 1.0\times$ (Naive wins by $5\times$)** |
| **$\ge 0.50$** | **200** | 500 | 500 | **$< 1.0\times$ (Naive wins by $2.5\times$)** |
| **$\ge 0.60$** | **200** | Not Reached | 1000 | **$< 1.0\times$ (Naive wins by $5\times$)** |
| **$\ge 0.70$** | **300** | Not Reached | Not Reached | **$< 1.0\times$ (Naive exclusive)** |
| **$\ge 0.75$** | **500** | Not Reached | Not Reached | **$< 1.0\times$ (Naive exclusive)** |

> [!IMPORTANT]
> No positive data-efficiency multiplier can be claimed for Intentra V2 over Naive Synthetic data on either TF-IDF or DistilBERT.

---

## 8. Exact Limitations of the Current Study

1. **Programmatic Fallback vs. Frontier LLM**:
   * The benchmark was executed in a standalone environment where `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` was not loaded, triggering the programmatic template synthesis fallback.
   * While programmatic templates accurately mimic boundary contrastive structures, frontier LLMs (e.g. Claude 3.5 Sonnet / GPT-4o) would generate more natural linguistic phrasing that may reduce raw token collision.
2. **Boundary-to-Canonical Ratio**:
   * In the current V2 implementation, targeted generation aggressively injects $40\%\text{--}50\%$ boundary/hard-negative data in closed-loop cycles. A curriculum-learning schedule (e.g., $90\%$ canonical initially, decaying to $15\%$ boundary data only after convergence) was not tested.

---

## 9. Recommended Next Experiment & Engineering Roadmap

To turn Intentra V2 into an empirically winning system:

1. **Curriculum Staged Data Generation**:
   * Ensure baseline datasets maintain $\ge 80\%$ canonical high-signal seed examples before injecting boundary cases.
2. **Quality Filter Keyword Cleansing**:
   * Update `core/quality_filter.py` to penalize synthetic examples that contain the literal label string of another class unless accompanied by explicit syntactic negation clauses.
3. **Frontier LLM Validation Run**:
   * Run a parallel benchmark with live API keys to isolate the effect of LLM semantic phrasing vs programmatic templates.

---
*Report certified and saved to `intentra_v2_transformer_audit_report.md`.*
