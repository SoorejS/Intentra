# Intentra V2.1 — Reproducibility Guide

This document details exact steps, environment requirements, and commands to reproduce all audit, benchmark, and ablation findings reported for Intentra V2.1.

---

## 1. Environment & Dependencies

```bash
# Python 3.11+ / Python 3.13 supported
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn`
- `torch` & `transformers`
- `fastapi` & `uvicorn`
- `sqlalchemy`
- `pytest`

---

## 2. Reproduction Steps

### A. Run Full Test Suite (14 Unit & Integration Tests)
```bash
pytest -v tests/
```
*Expected Output*: 14 passed in < 10 seconds.

### B. Run Full Subsystem Audit
```bash
python scratch/run_full_audit.py
```
*Produces*: `scratch/full_v2_audit.json` and `intentra_v2_system_audit.md`.

### C. Run Mechanistic Data Inspection
```bash
python scratch/inspect_v2_mechanisms.py
```
*Produces*: `scratch/v2_mechanistic_inspection.json`.  
*Confirms*: Canonical anchor rates, conflicting keyword rates, and lexical separability metrics.

### D. Run 8-Arm Scientific Ablation Benchmark
```bash
python scratch/run_v21_ablation_benchmark.py
```
*Evaluates*: 240 model runs (8 arms $\times$ 5 seeds $\times$ 6 budgets).  
*Produces*: `scratch/v21_raw_results.json` and displays the full comparison matrix.

### E. Run Independent Confirmation Experiment
```bash
python scratch/run_v21_confirmation.py
```
*Evaluates*: 7 brand-new unseen seeds across all 6 budgets.  
*Produces*: `scratch/v21_confirmation_results.json`.

---

## 3. Data Integrity & Isolation Guarantees

- **Locked Holdout Test Set ($D_{test}$)**: 50 balanced, hand-annotated boundary and hard-negative examples in `core/benchmark_suite.py` (`get_locked_holdout_test_set()`).
- **Dedicated Validation Set ($D_{val}$)**: 25 distinct examples in `core/benchmark_suite.py` (`get_dedicated_validation_set()`).
- **Zero Leakage**: All scripts programmatically assert `len(set(D_val).intersection(set(D_test))) == 0`.
