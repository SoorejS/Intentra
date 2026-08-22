"""
Intentra V2.1 - 8-Arm Scientific Ablation Benchmark Runner
Evaluates 8 experimental arms across 5 seeds x 6 sample sizes on locked holdout test set.

Ablation Arms:
  Arm A: Naive Synthetic (Direct keyword-separated baseline)
  Arm B: Intentra V1 (Static 50/20/30 mix)
  Arm C: Current Intentra V2 (Targeted boundary hardening without curriculum anchor floor)
  Arm D: Intentra V2.1 Curriculum (Full 4-stage progression with anchor coverage)
  Arm E: V2.1 w/o Boundary Examples (Stage 1+2+4: Anchors, variations, hard negatives)
  Arm F: V2.1 w/o Hard Negatives (Stage 1+2+3: Anchors, variations, boundaries)
  Arm G: V2.1 w/o Contrastive Cues (Stage 1+2+3 clean: No misleading keyword cues)
  Arm H: V2.1 Randomized Order (Same composition as D, but randomized order)

Evaluation Protocol:
  - 5 Deterministic Seeds: [42, 123, 456, 789, 999]
  - 6 Sample Budgets: [50, 100, 200, 300, 500, 1000]
  - Locked Holdout Test Set (50 examples)
  - Dedicated Validation Split (25 examples)
  - Zero leakage asserted and verified
"""

import os
import sys
import json
import time
import datetime
import numpy as np
from collections import defaultdict

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
from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model
from core.curriculum_scheduler import build_curriculum_dataset, CurriculumPolicy

SEEDS = [42, 123, 456, 789, 999]
SIZES = [50, 100, 200, 300, 500, 1000]

test_set = get_locked_holdout_test_set()
val_set = get_dedicated_validation_set()
schema = get_demo_customer_support_schema()
classes = [c["label"] for c in schema["output_classes"]]

# Assert zero leakage
test_texts = set(ex["text"] for ex in test_set)
val_texts = set(ex["text"] for ex in val_set)
assert len(test_texts.intersection(val_texts)) == 0, "Data leakage detected between Val and Test!"


# ── Dataset Builders for Ablation Arms ────────────────────────────────────────

def build_arm_d_dataset(count: int, seed: int = 42) -> list:
    """Arm D: Full V2.1 Curriculum (Dynamic stage 1 -> 4 progression)."""
    # Select stage based on budget maturity
    if count <= 100:
        stage = 2
    elif count <= 300:
        stage = 3
    else:
        stage = 4
    return build_curriculum_dataset(total_count=count, stage=stage, classes=classes, seed=seed)


def build_arm_e_dataset(count: int, seed: int = 42) -> list:
    """Arm E: V2.1 without Boundary Examples (Anchors + Variations + Hard Negatives)."""
    policy = CurriculumPolicy(min_canonical_ratio=0.50, max_boundary_ratio=0.0, max_hard_negative_ratio=0.25)
    return build_curriculum_dataset(total_count=count, stage=4, classes=classes, seed=seed, policy=policy)


def build_arm_f_dataset(count: int, seed: int = 42) -> list:
    """Arm F: V2.1 without Hard Negatives (Anchors + Variations + Boundaries)."""
    policy = CurriculumPolicy(min_canonical_ratio=0.50, max_boundary_ratio=0.35, max_hard_negative_ratio=0.0)
    return build_curriculum_dataset(total_count=count, stage=3, classes=classes, seed=seed, policy=policy)


def build_arm_g_dataset(count: int, seed: int = 42) -> list:
    """Arm G: V2.1 without Contrastive Examples (Stage 1 + 2 only: Pure anchors & variations)."""
    policy = CurriculumPolicy(min_canonical_ratio=0.60, max_boundary_ratio=0.0, max_hard_negative_ratio=0.0)
    return build_curriculum_dataset(total_count=count, stage=2, classes=classes, seed=seed, policy=policy)


def build_arm_h_dataset(count: int, seed: int = 42) -> list:
    """Arm H: V2.1 Data with Randomized Curriculum Order."""
    data = list(build_arm_d_dataset(count, seed=seed))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    return data


ARM_BUILDERS = {
    "arm_a_naive": lambda count, seed: build_naive_dataset(count, seed=seed),
    "arm_b_v1": lambda count, seed: build_v1_dataset(count, seed=seed),
    "arm_c_v2": lambda count, seed: build_v2_closed_loop_dataset(count, val_set=val_set, seed=seed)["dataset"],
    "arm_d_v21_curriculum": build_arm_d_dataset,
    "arm_e_v21_no_boundary": build_arm_e_dataset,
    "arm_f_v21_no_hard_neg": build_arm_f_dataset,
    "arm_g_v21_no_contrastive": build_arm_g_dataset,
    "arm_h_v21_randomized_order": build_arm_h_dataset,
}

ARM_NAMES = {
    "arm_a_naive": "Arm A: Naive Synthetic",
    "arm_b_v1": "Arm B: Intentra V1 (Static)",
    "arm_c_v2": "Arm C: Intentra V2 (Current)",
    "arm_d_v21_curriculum": "Arm D: Intentra V2.1 Curriculum (Full)",
    "arm_e_v21_no_boundary": "Arm E: V2.1 w/o Boundary Examples",
    "arm_f_v21_no_hard_neg": "Arm F: V2.1 w/o Hard Negatives",
    "arm_g_v21_no_contrastive": "Arm G: V2.1 w/o Contrastive",
    "arm_h_v21_randomized_order": "Arm H: V2.1 Randomized Order"
}

# ── Execute Full Benchmark Matrix ────────────────────────────────────────────

print("=" * 80)
print(f"INTENTRA V2.1 — 8-ARM EMPIRICAL ABLATION BENCHMARK ({len(SEEDS)} Seeds, {len(SIZES)} Budgets)")
print("=" * 80)

raw_results = {
    "benchmark_info": {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "framework": "sklearn_fast",
        "seeds": SEEDS,
        "sample_sizes": SIZES,
        "holdout_size": len(test_set),
        "validation_size": len(val_set),
        "arms": ARM_NAMES
    },
    "arms_data": {},
    "summary_matrix": {}
}

total_trainings = 0
start_time = time.time()

for arm_key, builder_fn in ARM_BUILDERS.items():
    arm_name = ARM_NAMES[arm_key]
    print(f"\n>>> Running Ablation: {arm_name}")
    raw_results["arms_data"][arm_key] = {}

    for size in SIZES:
        f1_list, acc_list, bnd_list, hn_list = [], [], [], []
        seed_fps = []
        seed_details = []

        for seed in SEEDS:
            data = builder_fn(size, seed)
            fp = compute_dataset_fingerprint(data)
            seed_fps.append(fp)

            trainer = train_classifier(data, framework="sklearn_fast", seed=seed)
            eval_res = evaluate_model(trainer["predictor"], test_set)
            total_trainings += 1

            f1 = eval_res["macro_f1"]
            acc = eval_res["accuracy"]
            bnd = eval_res["boundary_accuracy"]
            hn = eval_res["hard_negative_accuracy"]

            f1_list.append(f1)
            acc_list.append(acc)
            bnd_list.append(bnd)
            hn_list.append(hn)

            seed_details.append({
                "seed": seed,
                "dataset_fingerprint": fp,
                "macro_f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "boundary_accuracy": round(bnd, 4),
                "hard_negative_accuracy": round(hn, 4)
            })

        # Rigorous independence assertion: Fail if seeds produced identical fingerprints!
        assert len(set(seed_fps)) == len(SEEDS), (
            f"FATAL: False seed independence in {arm_key} at size {size}! "
            f"Unique fingerprints: {len(set(seed_fps))} / {len(SEEDS)}"
        )

        mean_f1 = float(np.mean(f1_list))
        std_f1 = float(np.std(f1_list))
        mean_acc = float(np.mean(acc_list))
        mean_bnd = float(np.mean(bnd_list))
        mean_hn = float(np.mean(hn_list))

        raw_results["arms_data"][arm_key][size] = {
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "mean_accuracy": round(mean_acc, 4),
            "mean_boundary_accuracy": round(mean_bnd, 4),
            "mean_hard_negative_accuracy": round(mean_hn, 4),
            "seed_runs_f1": [round(x, 4) for x in f1_list],
            "seed_runs_acc": [round(x, 4) for x in acc_list],
            "seed_details": seed_details
        }

        print(f"  * N={size:4d}: Mean Macro F1 = {mean_f1:.4f} ± {std_f1:.4f} | Bnd Acc: {mean_bnd:.4f} | HN Acc: {mean_hn:.4f}")

elapsed_total = round(time.time() - start_time, 2)
print(f"\n[Completed] Evaluated {total_trainings} models in {elapsed_total}s.")

# ── Summary Matrix Comparison Table ──────────────────────────────────────────
summary_table = {}
for size in SIZES:
    summary_table[str(size)] = {
        arm_key: raw_results["arms_data"][arm_key][size]["mean_macro_f1"]
        for arm_key in ARM_BUILDERS.keys()
    }

raw_results["summary_matrix"] = summary_table
raw_results["telemetry"] = {
    "total_models_trained": total_trainings,
    "elapsed_seconds": elapsed_total
}

# Write raw results to scratch/v21_raw_results.json
with open(os.path.join(root_dir, "scratch", "v21_raw_results.json"), "w", encoding="utf-8") as f:
    json.dump(raw_results, f, indent=2)

print("\nSaved raw machine-readable results to scratch/v21_raw_results.json!")
