"""
Intentra V2 - Evaluation Engine
Calculates standard classification metrics (Macro F1, Accuracy, Precision, Recall),
per-class metrics, confusion matrices, and granular slice performance
(Boundary, Hard-Negative, and Adversarial slice accuracy).
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_model(predictor, test_dataset: list) -> dict:
    """
    Evaluate a trained model predictor against a test/validation dataset.
    test_dataset: list of dicts with {"text": str, "label": str, Optional("type"): str}
    """
    if not test_dataset:
        raise ValueError("Cannot evaluate on empty test dataset")

    texts = [item["text"] for item in test_dataset]
    ground_truth = [item["label"] for item in test_dataset]
    types = [item.get("type", "standard") for item in test_dataset]

    predictions, confidences = predictor.predict(texts)

    # 1. Overall Metrics
    classes = sorted(list(set(ground_truth + predictions)))
    acc = float(accuracy_score(ground_truth, predictions))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="weighted", zero_division=0
    )

    # 2. Per-Class Metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        ground_truth, predictions, labels=classes, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, cls in enumerate(classes):
        per_class_metrics[cls] = {
            "precision": round(float(prec_per_class[i]), 4),
            "recall": round(float(rec_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": int(support_per_class[i])
        }

    # 3. Confusion Matrix
    cm_array = confusion_matrix(ground_truth, predictions, labels=classes)
    confusion_matrix_dict = {
        "classes": classes,
        "matrix": cm_array.tolist()
    }

    # 4. Granular Slice Accuracies (Boundary, Hard-Negative, Adversarial)
    def compute_slice_accuracy(target_type):
        slice_indices = [i for i, t in enumerate(types) if t == target_type]
        if not slice_indices:
            return 0.0
        slice_true = [ground_truth[i] for i in slice_indices]
        slice_pred = [predictions[i] for i in slice_indices]
        return round(float(accuracy_score(slice_true, slice_pred)), 4)

    boundary_acc = compute_slice_accuracy("boundary")
    hard_neg_acc = compute_slice_accuracy("hard_negative")
    adversarial_acc = compute_slice_accuracy("adversarial")

    # 5. Raw Detailed Output
    detailed_results = []
    for i in range(len(test_dataset)):
        detailed_results.append({
            "text": texts[i],
            "expected_label": ground_truth[i],
            "predicted_label": predictions[i],
            "confidence": round(float(confidences[i]), 4) if i < len(confidences) else 1.0,
            "type": types[i],
            "is_correct": ground_truth[i] == predictions[i]
        })

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "precision": round(float(prec_macro), 4),
        "recall": round(float(rec_macro), 4),
        "boundary_accuracy": boundary_acc,
        "hard_negative_accuracy": hard_neg_acc,
        "adversarial_accuracy": adversarial_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix_dict,
        "detailed_results": detailed_results,
        "total_evaluated": len(test_dataset)
    }
