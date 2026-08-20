"""
Intentra V2 - Phase 1 Transformer (DistilBERT) Validation Benchmark
Executes 5 seeds x 6 sample sizes x 3 methods using DistilBERT Sequence Classification.
Preserves:
  - Exact locked holdout test set (50 examples)
  - Exact validation set (25 examples)
  - Exact training sample generation pools (Naive, V1, V2)
  - Zero leakage (|D_val ∩ D_test| = 0)
"""

import os
import sys
import json
import time
import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

torch.set_num_threads(os.cpu_count() or 4)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from core.evaluation_engine import evaluate_model
from core.classifier_trainer import TransformerPredictor
from core.benchmark_suite import (
    get_locked_holdout_test_set,
    get_dedicated_validation_set,
    get_demo_customer_support_schema,
    build_naive_dataset,
    build_v1_dataset,
    build_v2_closed_loop_dataset
)

SEEDS = [42, 123, 456, 789, 999]
SIZES = [50, 100, 200, 300, 500, 1000]
METHODS = ["naive", "intentra_v1", "intentra_v2"]

test_set = get_locked_holdout_test_set()
val_set = get_dedicated_validation_set()
schema = get_demo_customer_support_schema()

# Verification of zero leakage
test_texts = set(ex["text"] for ex in test_set)
val_texts = set(ex["text"] for ex in val_set)
assert len(test_texts.intersection(val_texts)) == 0, "Data leakage detected between Val and Test!"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def train_distilbert_instance(dataset, seed, epochs=3, lr=5e-5, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels = sorted(list(set(item["label"] for item in dataset)))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Freeze lower layers for efficient CPU fine-tuning
    for param in model.distilbert.transformer.layer[:4].parameters():
        param.requires_grad = False

    numeric_labels = [label2id[item["label"]] for item in dataset]
    hf_data = HFDataset.from_dict({"text": [item["text"] for item in dataset], "label": numeric_labels})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

    tokenized = hf_data.map(tokenize_fn, batched=True)

    out_dir = f"./scratch/tmp_distilbert_run_{seed}_{int(time.time()*1000)%100000}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

    predictor = TransformerPredictor(model, tokenizer, label2id, id2label, device="cpu")
    
    # Cleanup temp checkpoint artifacts
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)

    return predictor


print("="*70)
print(f"STARTING DISTILBERT VALIDATION MATRIX ({len(SEEDS)} Seeds x {len(SIZES)} Budgets x 3 Methods)")
print("="*70)

distilbert_results = {
    "task": "Customer Support Intent Classification (5 Classes)",
    "classifier": "DistilBERT (distilbert-base-uncased)",
    "seeds": SEEDS,
    "sample_sizes": SIZES,
    "holdout_test_set_size": len(test_set),
    "dedicated_validation_set_size": len(val_set),
    "data_leakage_detected": False,
    "by_method": {
        "naive": {},
        "intentra_v1": {},
        "intentra_v2": {}
    },
    "generator_quality_audit": {
        "rejection_counts_by_size": {},
        "observed_confusion_pairs": [],
        "sample_generated_boundary_examples": []
    },
    "telemetry": {
        "total_runs": 0,
        "total_time_seconds": 0.0
    }
}

start_matrix_time = time.time()
total_runs = 0

for size in SIZES:
    print(f"\n--- Benchmarking Sample Budget: {size} Examples ---")

    for method in METHODS:
        f1s, accs, bnd_accs, hn_accs, per_class_list = [], [], [], [], []

        for seed in SEEDS:
            t_run_start = time.time()
            if method == "naive":
                dataset = build_naive_dataset(size, seed=seed)
            elif method == "intentra_v1":
                dataset = build_v1_dataset(size, seed=seed)
            elif method == "intentra_v2":
                v2_obj = build_v2_closed_loop_dataset(size, val_set=val_set, seed=seed, framework="sklearn_fast")
                dataset = v2_obj["dataset"]
                if seed == 42 and size == 100:
                    distilbert_results["generator_quality_audit"]["rejection_counts_by_size"][size] = v2_obj["telemetry"]["rejections"]
                    distilbert_results["generator_quality_audit"]["observed_confusion_pairs"] = v2_obj["diagnostics"].get("top_confused_pairs", [])
                    distilbert_results["generator_quality_audit"]["sample_generated_boundary_examples"] = dataset[-5:]

            predictor = train_distilbert_instance(dataset, seed=seed)
            eval_res = evaluate_model(predictor, test_set)
            run_dur = time.time() - t_run_start
            total_runs += 1

            f1s.append(eval_res["macro_f1"])
            accs.append(eval_res["accuracy"])
            bnd_accs.append(eval_res["boundary_accuracy"])
            hn_accs.append(eval_res["hard_negative_accuracy"])
            per_class_list.append(eval_res["per_class_metrics"])

            print(f"  [{method.upper()}] Seed {seed} (N={size}): Macro F1 = {eval_res['macro_f1']:.4f}, Acc = {eval_res['accuracy']:.4f} ({run_dur:.1f}s)")

        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        ci95_f1 = float(1.96 * (std_f1 / np.sqrt(len(f1s))))

        distilbert_results["by_method"][method][size] = {
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "ci95_macro_f1": round(ci95_f1, 4),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "mean_boundary_accuracy": round(float(np.mean(bnd_accs)), 4),
            "mean_hard_negative_accuracy": round(float(np.mean(hn_accs)), 4),
            "raw_f1_runs": [round(float(x), 4) for x in f1s]
        }

total_matrix_time = time.time() - start_matrix_time
distilbert_results["telemetry"]["total_runs"] = total_runs
distilbert_results["telemetry"]["total_time_seconds"] = round(total_matrix_time, 2)

print("\n" + "="*70)
print(f"DISTILBERT BENCHMARK COMPLETED IN {total_matrix_time:.2f}s ({total_runs} RUNS)")
print("="*70)

with open(os.path.join(root_dir, "scratch", "distilbert_raw_data.json"), "w", encoding="utf-8") as f:
    json.dump(distilbert_results, f, indent=2)

print("Saved raw output to scratch/distilbert_raw_data.json!")
