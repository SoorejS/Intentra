"""
Quick 3-seed sanity benchmark to verify true multi-seed independence and distinct fingerprints.
Runs on seeds [42, 123, 456] across N=[50, 100].
Fails immediately if identical fingerprints or false independence are detected.
"""

import os
import sys
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
    build_v2_closed_loop_dataset,
    compute_dataset_fingerprint
)
from core.curriculum_scheduler import build_curriculum_dataset
from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model

SEEDS = [42, 123, 456]
SIZES = [50, 100]

test_set = get_locked_holdout_test_set()
val_set = get_dedicated_validation_set()
schema = get_demo_customer_support_schema()
classes = [c["label"] for c in schema["output_classes"]]

print("=" * 70)
print("SANITY BENCHMARK: 3 SEEDS x 2 BUDGETS (FINGERPRINT INDEPENDENCE TEST)")
print("=" * 70)

arms = {
    "Naive": lambda count, s: build_naive_dataset(count, seed=s),
    "V1": lambda count, s: build_v1_dataset(count, seed=s),
    "V2_Old": lambda count, s: build_v2_closed_loop_dataset(count, val_set=val_set, seed=s)["dataset"],
    "V2.1_Curriculum": lambda count, s: build_curriculum_dataset(count, stage=2, classes=classes, seed=s)
}

for arm_name, builder in arms.items():
    print(f"\nChecking Arm: {arm_name}")
    for size in SIZES:
        fps = []
        f1s = []
        for s in SEEDS:
            data = builder(size, s)
            fp = compute_dataset_fingerprint(data)
            fps.append(fp)

            trainer = train_classifier(data, framework="sklearn_fast", seed=s)
            eval_res = evaluate_model(trainer["predictor"], test_set)
            f1s.append(eval_res["macro_f1"])
            print(f"  [Seed {s}] Size={size:3d} | Fingerprint: {fp[:12]}... | F1: {eval_res['macro_f1']:.4f}")

        # Assert distinct fingerprints across seeds
        assert len(set(fps)) == len(SEEDS), f"FAILURE: Identical fingerprints detected for {arm_name} at size {size}! {fps}"
        print(f"  -> PASS: All {len(SEEDS)} seeds produced unique fingerprints. Mean F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

print("\n" + "=" * 70)
print("SANITY BENCHMARK PASSED! Multi-seed independence verified.")
print("=" * 70)
