"""
Intentra V2 - Error Analyzer Module
Analyzes evaluation results to uncover weak classes, confused decision boundary pairs,
and categorizes classification error types to guide targeted synthetic data generation.
"""

from collections import defaultdict


def analyze_errors(evaluation_results: dict) -> dict:
    """
    Given evaluation results from evaluation_engine, extract:
    - Ranked weak classes
    - Confused class pairs (CA <-> CB)
    - Error breakdowns and categorizations
    - Targeted diagnostic problem summary
    """
    per_class = evaluation_results.get("per_class_metrics", {})
    cm = evaluation_results.get("confusion_matrix", {})
    detailed = evaluation_results.get("detailed_results", [])

    # 1. Rank weakest classes by F1
    weakest_classes = []
    for cls, metrics in per_class.items():
        weakest_classes.append({
            "class_name": cls,
            "f1": metrics.get("f1", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "support": metrics.get("support", 0)
        })
    weakest_classes.sort(key=lambda x: x["f1"])

    # 2. Extract Top Confused Pairs from Confusion Matrix
    classes = cm.get("classes", [])
    matrix = cm.get("matrix", [])
    pair_counts = defaultdict(int)
    directional_errors = defaultdict(int)

    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and i < len(matrix) and j < len(matrix[i]):
                count = matrix[i][j]
                if count > 0:
                    pair_key = tuple(sorted([true_cls, pred_cls]))
                    pair_counts[pair_key] += count
                    directional_errors[(true_cls, pred_cls)] += count

    confused_pairs = []
    for (cls_a, cls_b), total_errs in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        a_to_b = directional_errors.get((cls_a, cls_b), 0)
        b_to_a = directional_errors.get((cls_b, cls_a), 0)
        confused_pairs.append({
            "class_a": cls_a,
            "class_b": cls_b,
            "total_mutual_errors": total_errs,
            "a_predicted_as_b": a_to_b,
            "b_predicted_as_a": b_to_a,
            "severity": "HIGH" if total_errs >= 5 else "MEDIUM" if total_errs >= 2 else "LOW"
        })

    # 3. Categorize Individual Classification Errors
    errors_categorized = []
    error_counts_by_type = defaultdict(int)

    for item in detailed:
        if not item.get("is_correct", False):
            expected = item.get("expected_label")
            predicted = item.get("predicted_label")
            conf = item.get("confidence", 0.0)
            item_type = item.get("type", "standard")

            # Determine error type
            if item_type == "hard_negative":
                error_type = "hard_negative_failure"
            elif item_type == "boundary":
                error_type = "boundary_failure"
            elif item_type == "adversarial":
                error_type = "adversarial_failure"
            elif conf < 0.55:
                error_type = "low_confidence"
            else:
                error_type = "class_confusion"

            error_counts_by_type[error_type] += 1
            errors_categorized.append({
                "input_text": item.get("text"),
                "expected_label": expected,
                "predicted_label": predicted,
                "confidence": conf,
                "error_type": error_type,
                "source_type": item_type
            })

    # 4. Synthesize Diagnostic Problem Summary
    if confused_pairs:
        top_pair = confused_pairs[0]
        top_weak = weakest_classes[0] if weakest_classes else None
        target_problem = (
            f"Severe decision boundary confusion between '{top_pair['class_a']}' and '{top_pair['class_b']}' "
            f"({top_pair['total_mutual_errors']} misclassifications). "
            f"Lowest performing class: '{top_weak['class_name']}' (F1: {top_weak['f1']:.3f})."
        )
    elif weakest_classes:
        top_weak = weakest_classes[0]
        target_problem = f"Weakest class representation: '{top_weak['class_name']}' with F1 of {top_weak['f1']:.3f}."
    else:
        target_problem = "No critical misclassifications detected."

    return {
        "weakest_classes": weakest_classes,
        "confused_pairs": confused_pairs,
        "total_errors": len(errors_categorized),
        "error_breakdown_by_type": dict(error_counts_by_type),
        "categorized_errors": errors_categorized,
        "target_problem_summary": target_problem,
        "primary_focus_pair": (confused_pairs[0]["class_a"], confused_pairs[0]["class_b"]) if confused_pairs else None,
        "primary_focus_weak_class": weakest_classes[0]["class_name"] if weakest_classes else None
    }
