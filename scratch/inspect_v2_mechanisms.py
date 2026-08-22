"""
Intentra V2 - Phase 3 Mechanistic Inspection of V2 Data
Quantitatively measures dataset properties (archetype rates, lexical overlap,
conflicting keyword contamination, class balance) across Naive, V1, and V2 datasets.
Outputs scratch/v2_mechanistic_inspection.json.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.benchmark_suite import (
    build_naive_dataset,
    build_v1_dataset,
    build_v2_closed_loop_dataset,
    get_dedicated_validation_set,
    get_demo_customer_support_schema
)

# Class signature keywords for Customer Support schema
CLASS_KEYWORDS = {
    "refund_request": {"refund", "money", "reimbursement", "charge", "return", "funds", "credit", "reimburse"},
    "cancellation_request": {"cancel", "cancellation", "terminate", "abort", "discontinue", "stop", "void", "revoke"},
    "billing_inquiry": {"billing", "invoice", "fee", "receipt", "tax", "vat", "prorated", "statement", "payment"},
    "technical_support": {"bug", "crash", "error", "403", "500", "ssl", "sso", "login", "password", "screen", "freeze"},
    "general_feedback": {"feedback", "design", "ui", "ux", "team", "feature", "product", "experience", "suggestion"}
}

def tokenize(text):
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())

def measure_dataset_mechanics(dataset, name="Dataset"):
    total = len(dataset)
    if total == 0:
        return {}

    # 1. Archetype rates
    type_counts = Counter(item.get("type", "unknown") for item in dataset)
    canonical_rate = type_counts["canonical"] / total
    boundary_rate = type_counts["boundary"] / total
    hard_neg_rate = type_counts["hard_negative"] / total
    adversarial_rate = (type_counts["adversarial"] + type_counts["contrastive"]) / total

    # 2. Class balance
    label_counts = Counter(item.get("label") for item in dataset)
    classes = list(label_counts.keys())
    ideal_count = total / len(classes) if classes else 1
    balance_std = float(np.std([label_counts[c] for c in classes]))

    # 3. Conflicting keyword contamination
    # Check if text contains strong keywords belonging to other classes
    conflicting_count = 0
    cross_class_keyword_matches = defaultdict(int)

    for item in dataset:
        lbl = item.get("label")
        tokens = tokenize(item.get("text", ""))
        has_conflict = False
        for other_cls, kws in CLASS_KEYWORDS.items():
            if other_cls != lbl:
                overlap = tokens.intersection(kws)
                if len(overlap) >= 2 or (len(overlap) >= 1 and item.get("type") in ["hard_negative", "adversarial"]):
                    has_conflict = True
                    cross_class_keyword_matches[(lbl, other_cls)] += 1
        if has_conflict:
            conflicting_count += 1

    conflicting_rate = conflicting_count / total

    # 4. Lexical overlap (Inter-class vs Intra-class Jaccard similarity)
    intra_jaccards = []
    inter_jaccards = []

    # Sample pairwise for speed if large
    sample_items = dataset[:min(100, total)]
    for i in range(len(sample_items)):
        tokens_i = tokenize(sample_items[i]["text"])
        for j in range(i + 1, len(sample_items)):
            tokens_j = tokenize(sample_items[j]["text"])
            if not tokens_i or not tokens_j:
                continue
            jacc = len(tokens_i.intersection(tokens_j)) / len(tokens_i.union(tokens_j))
            if sample_items[i]["label"] == sample_items[j]["label"]:
                intra_jaccards.append(jacc)
            else:
                inter_jaccards.append(jacc)

    avg_intra_jaccard = float(np.mean(intra_jaccards)) if intra_jaccards else 0.0
    avg_inter_jaccard = float(np.mean(inter_jaccards)) if inter_jaccards else 0.0
    separability_ratio = (avg_intra_jaccard / (avg_inter_jaccard + 1e-6))

    return {
        "dataset_name": name,
        "total_examples": total,
        "canonical_anchor_rate": round(canonical_rate, 4),
        "boundary_example_rate": round(boundary_rate, 4),
        "hard_negative_rate": round(hard_neg_rate, 4),
        "adversarial_contrastive_rate": round(adversarial_rate, 4),
        "conflicting_keyword_rate": round(conflicting_rate, 4),
        "class_balance_std": round(balance_std, 4),
        "avg_intra_class_jaccard": round(avg_intra_jaccard, 4),
        "avg_inter_class_jaccard": round(avg_inter_jaccard, 4),
        "lexical_separability_ratio": round(separability_ratio, 4),
        "type_counts": dict(type_counts),
        "label_counts": dict(label_counts)
    }

print("Running mechanistic measurements on N=100 datasets across methods...")
val_set = get_dedicated_validation_set()

naive_data = build_naive_dataset(100, seed=42)
v1_data = build_v1_dataset(100, seed=42)
v2_obj = build_v2_closed_loop_dataset(100, val_set=val_set, seed=42, framework="sklearn_fast")
v2_data = v2_obj["dataset"]

naive_metrics = measure_dataset_mechanics(naive_data, "Naive Synthetic")
v1_metrics = measure_dataset_mechanics(v1_data, "Intentra V1")
v2_metrics = measure_dataset_mechanics(v2_data, "Intentra V2 Closed-Loop")

# Test Hypothesis
# Premature boundary hardening hypothesis: V2 has low canonical anchor rate, high conflicting keyword rate, and low lexical separability.
hypothesis_supported = (
    v2_metrics["canonical_anchor_rate"] < naive_metrics["canonical_anchor_rate"] and
    v2_metrics["conflicting_keyword_rate"] > naive_metrics["conflicting_keyword_rate"] and
    v2_metrics["lexical_separability_ratio"] < naive_metrics["lexical_separability_ratio"]
)

results = {
    "naive": naive_metrics,
    "intentra_v1": v1_metrics,
    "intentra_v2": v2_metrics,
    "hypothesis_evaluation": {
        "hypothesis": "Premature Boundary Hardening (excessive boundary/contrastive examples before canonical anchors are established)",
        "supported": bool(hypothesis_supported),
        "evidence": {
            "canonical_rate_comparison": f"Naive: {naive_metrics['canonical_anchor_rate']} vs V2: {v2_metrics['canonical_anchor_rate']}",
            "conflicting_keyword_rate_comparison": f"Naive: {naive_metrics['conflicting_keyword_rate']} vs V2: {v2_metrics['conflicting_keyword_rate']}",
            "lexical_separability_ratio_comparison": f"Naive: {naive_metrics['lexical_separability_ratio']} vs V2: {v2_metrics['lexical_separability_ratio']}"
        },
        "conclusion": "SUPPORTED" if hypothesis_supported else "NOT_SUPPORTED"
    }
}

with open(os.path.join(root_dir, "scratch", "v2_mechanistic_inspection.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n--- Mechanistic Inspection Results ---")
print(f"Naive: Canonical={naive_metrics['canonical_anchor_rate']}, Conflicting={naive_metrics['conflicting_keyword_rate']}, Separability={naive_metrics['lexical_separability_ratio']}")
print(f"V1:    Canonical={v1_metrics['canonical_anchor_rate']}, Conflicting={v1_metrics['conflicting_keyword_rate']}, Separability={v1_metrics['lexical_separability_ratio']}")
print(f"V2:    Canonical={v2_metrics['canonical_anchor_rate']}, Conflicting={v2_metrics['conflicting_keyword_rate']}, Separability={v2_metrics['lexical_separability_ratio']}")
print(f"\nHypothesis verdict: {results['hypothesis_evaluation']['conclusion']}")
