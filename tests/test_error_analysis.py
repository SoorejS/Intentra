"""
Unit Tests for Intentra V2 Error Analyzer Module.
"""

from core.error_analyzer import analyze_errors


def test_error_analyzer_detects_weak_classes_and_confusion_pairs():
    mock_eval = {
        "per_class_metrics": {
            "refund_request": {"f1": 0.85, "precision": 0.85, "recall": 0.85, "support": 20},
            "cancel_order": {"f1": 0.60, "precision": 0.60, "recall": 0.60, "support": 20},
            "billing_help": {"f1": 0.95, "precision": 0.95, "recall": 0.95, "support": 20}
        },
        "confusion_matrix": {
            "classes": ["billing_help", "cancel_order", "refund_request"],
            "matrix": [
                [19, 1, 0],
                [0, 12, 8],
                [0, 3, 17]
            ]
        },
        "detailed_results": [
            {"text": "Cancel but refund me", "expected_label": "cancel_order", "predicted_label": "refund_request", "confidence": 0.65, "type": "boundary", "is_correct": False},
            {"text": "I want money back", "expected_label": "refund_request", "predicted_label": "cancel_order", "confidence": 0.70, "type": "hard_negative", "is_correct": False},
        ]
    }

    diag = analyze_errors(mock_eval)

    assert len(diag["weakest_classes"]) == 3
    assert diag["weakest_classes"][0]["class_name"] == "cancel_order" # Lowest F1 (0.60)
    assert diag["primary_focus_weak_class"] == "cancel_order"

    assert len(diag["confused_pairs"]) > 0
    top_pair = diag["confused_pairs"][0]
    assert top_pair["class_a"] == "cancel_order"
    assert top_pair["class_b"] == "refund_request"
    assert top_pair["total_mutual_errors"] == 11 # 8 + 3
    assert "cancel_order" in diag["target_problem_summary"]
    assert "refund_request" in diag["target_problem_summary"]
