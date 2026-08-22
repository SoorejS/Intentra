"""
Intentra V2.1 - Independent Confirmation Benchmark
Runs 7 unseen random seeds across all 6 budgets on the locked holdout test set
to independently confirm statistical significance and generalizability of the V2.1 curriculum.
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
    get_locked_holdout_test_set,
    get_dedicated_validation_set,
    get_demo_customer_support_schema,
    build_naive_dataset,
    build_v1_dataset,
    build_v2_closed_loop_dataset
)
from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model
from core.curriculum_scheduler import build_curriculum_dataset

CONFIRMATION_SEEDS = [111, 222, 333, 444, 555, 777, 888]
SIZES = [50, 100, 200, 300, 500, 1000]

test_set = get_locked_holdout_test_set()
val_set = get_dedicated_validation_set()
schema = get_demo_customer_support_schema()
classes = [c["label"] for c in schema["output_classes"]]

def build_v21(count, seed):
    stage = 2 if count <= 100 else 3 if count <= 300 else 4
    return build_curriculum_dataset(total_count=count, stage=stage, classes=classes, seed=seed)

print("=" * 80)
print(f"INTENTRA V2.1 — INDEPENDENT CONFIRMATION EXPERIMENT ({len(CONFIRMATION_SEEDS)} New Seeds)")
print("=" * 80)

confirmation_results = {
    "seeds": CONFIRMATION_SEEDS,
    "sizes": SIZES,
    "naive": {},
    "v1": {},
    "v2_old": {},
    "v21_curriculum": {}
}

for size in SIZES:
    print(f"\n[Evaluating Sample Budget: {size} Examples across {len(CONFIRMATION_SEEDS)} Seeds]")
    naive_f1s, v1_f1s, v2_f1s, v21_f1s = [], [], [], []
    v21_bnds, v21_hns = [], []

    for seed in CONFIRMATION_SEEDS:
        # Naive
        d_naive = build_naive_dataset(size, seed=seed)
        t_naive = train_classifier(d_naive, framework="sklearn_fast", seed=seed)
        e_naive = evaluate_model(t_naive["predictor"], test_set)
        naive_f1s.append(e_naive["macro_f1"])

        # V1
        d_v1 = build_v1_dataset(size, seed=seed)
        t_v1 = train_classifier(d_v1, framework="sklearn_fast", seed=seed)
        e_v1 = evaluate_model(t_v1["predictor"], test_set)
        v1_f1s.append(e_v1["macro_f1"])

        # V2 Old
        d_v2 = build_v2_closed_loop_dataset(size, val_set=val_set, seed=seed)["dataset"]
        t_v2 = train_classifier(d_v2, framework="sklearn_fast", seed=seed)
        e_v2 = evaluate_model(t_v2["predictor"], test_set)
        v2_f1s.append(e_v2["macro_f1"])

        # V2.1 Curriculum
        d_v21 = build_v21(size, seed=seed)
        t_v21 = train_classifier(d_v21, framework="sklearn_fast", seed=seed)
        e_v21 = evaluate_model(t_v21["predictor"], test_set)
        v21_f1s.append(e_v21["macro_f1"])
        v21_bnds.append(e_v21["boundary_accuracy"])
        v21_hns.append(e_v21["hard_negative_accuracy"])

    confirmation_results["naive"][size] = {"mean": round(float(np.mean(naive_f1s)), 4), "std": round(float(np.std(naive_f1s)), 4)}
    confirmation_results["v1"][size] = {"mean": round(float(np.mean(v1_f1s)), 4), "std": round(float(np.std(v1_f1s)), 4)}
    confirmation_results["v2_old"][size] = {"mean": round(float(np.mean(v2_f1s)), 4), "std": round(float(np.std(v2_f1s)), 4)}
    confirmation_results["v21_curriculum"][size] = {
        "mean": round(float(np.mean(v21_f1s)), 4),
        "std": round(float(np.std(v21_f1s)), 4),
        "mean_boundary_acc": round(float(np.mean(v21_bnds)), 4),
        "mean_hard_neg_acc": round(float(np.mean(v21_hns)), 4)
    }

    print(f"  * Naive:          Mean Macro F1 = {np.mean(naive_f1s):.4f} ± {np.std(naive_f1s):.4f}")
    print(f"  * V1:             Mean Macro F1 = {np.mean(v1_f1s):.4f} ± {np.std(v1_f1s):.4f}")
    print(f"  * V2 (Old):       Mean Macro F1 = {np.mean(v2_f1s):.4f} ± {np.std(v2_f1s):.4f}")
    print(f"  * V2.1 (Curric):  Mean Macro F1 = {np.mean(v21_f1s):.4f} ± {np.std(v21_f1s):.4f} | Bnd: {np.mean(v21_bnds):.4f} | HN: {np.mean(v21_hns):.4f}")

with open(os.path.join(root_dir, "scratch", "v21_confirmation_results.json"), "w", encoding="utf-8") as f:
    json.dump(confirmation_results, f, indent=2)

print("\nConfirmation results successfully saved to scratch/v21_confirmation_results.json!")
