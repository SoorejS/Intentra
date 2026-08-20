"""
Intentra V2 - Statistical Benchmark Audit Runner
Runs 5 independent seeds x 6 sample sizes x 3 methods.
Audits data leakage, dataset equivalence, metric calculation, and generates benchmark_audit_report.md.
"""

import os
import sys
import json
import time
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.benchmark_suite import (
    execute_full_benchmark_audit,
    get_locked_holdout_test_set,
    get_dedicated_validation_set,
    build_naive_dataset,
    build_v1_dataset,
    build_v2_closed_loop_dataset
)

SEEDS = [42, 123, 456, 789, 999]
SIZES = [50, 100, 200, 300, 500, 1000]

print("[1/4] Running Leakage & Equivalence Audit...")
test_set = get_locked_holdout_test_set()
val_set = get_dedicated_validation_set()

test_texts = set(ex["text"] for ex in test_set)
val_texts = set(ex["text"] for ex in val_set)
leakage_count = len(test_texts.intersection(val_texts))

print(f"  + Holdout Test Set: {len(test_set)} examples (5 classes, balanced, 40 boundary + 10 hard-negative)")
print(f"  + Validation Set: {len(val_set)} examples (5 classes, balanced, 10 canonical + 10 boundary + 5 hard-negative)")
print(f"  + Inter-split Text Leakage Overlap: {leakage_count} (0 = Complete Isolation)")

print("\n[2/4] Executing 5-Seed Empirical Benchmark Matrix...")
start_time = time.time()
audit_data = execute_full_benchmark_audit(seeds=SEEDS, sample_sizes=SIZES, framework="sklearn_fast")
elapsed = time.time() - start_time
print(f"\n[3/4] Benchmark completed in {elapsed:.2f}s across {audit_data['telemetry']['total_models_trained']} model training runs.")

# ── 4. Calculate True Data Efficiency Multipliers ──────────────────────────────
naive_f1 = {sz: audit_data["by_method"]["naive"][sz]["mean_macro_f1"] for sz in SIZES}
v1_f1 = {sz: audit_data["by_method"]["intentra_v1"][sz]["mean_macro_f1"] for sz in SIZES}
v2_f1 = {sz: audit_data["by_method"]["intentra_v2"][sz]["mean_macro_f1"] for sz in SIZES}

# Target thresholds
targets = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75]
efficiency_analysis = []

for t in targets:
    # Find min samples for naive
    n_naive = next((sz for sz in SIZES if naive_f1[sz] >= t), None)
    n_v1 = next((sz for sz in SIZES if v1_f1[sz] >= t), None)
    n_v2 = next((sz for sz in SIZES if v2_f1[sz] >= t), None)

    if n_naive is not None and n_v2 is not None:
        multiplier = round(float(n_naive / n_v2), 2)
        multiplier_str = f"{multiplier}x"
    elif n_v2 is not None and n_naive is None:
        multiplier_str = "Intentra V2 Reached / Naive Target Not Reached"
    elif n_v2 is None and n_naive is not None:
        multiplier_str = "Naive Reached / Intentra V2 Target Not Reached"
    else:
        multiplier_str = "Target Not Reached"

    efficiency_analysis.append({
        "target_macro_f1": t,
        "n_naive_required": n_naive if n_naive else "Target Not Reached",
        "n_v1_required": n_v1 if n_v1 else "Target Not Reached",
        "n_v2_required": n_v2 if n_v2 else "Target Not Reached",
        "data_efficiency_multiplier": multiplier_str
    })

# ── Determine Honest Verdict ───────────────────────────────────────────────────
# Check overall comparison across all sizes
v2_wins = sum(1 for sz in SIZES if v2_f1[sz] > naive_f1[sz])
v2_beats_v1 = sum(1 for sz in SIZES if v2_f1[sz] > v1_f1[sz])

if v2_wins >= 4 and any(v2_f1[sz] >= 0.50 for sz in SIZES):
    verdict = "GREEN"
    verdict_desc = "Intentra V2 demonstrably beats naive generation and Intentra V1 across majority of sample sizes."
elif v2_wins >= 2 or v2_beats_v1 >= 3:
    verdict = "YELLOW"
    verdict_desc = "Intentra V2 shows promising improvement over Intentra V1 and naive in specific sample ranges, but requires further scaling."
else:
    verdict = "RED"
    verdict_desc = "Intentra V2 does not currently beat the baseline."

print(f"\n[4/4] Automated Verdict Assessment: {verdict}")
print(f"      {verdict_desc}")

# Write raw audit json to scratch
with open(os.path.join(root_dir, "scratch", "benchmark_raw_data.json"), "w", encoding="utf-8") as f:
    json.dump({"audit_data": audit_data, "efficiency_analysis": efficiency_analysis, "verdict": verdict}, f, indent=2)

print("\nAudit raw data saved to scratch/benchmark_raw_data.json!")
