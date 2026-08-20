"""
Unit Tests for Intentra V2 Evaluation Engine.
"""

from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model


def test_evaluation_engine_metrics():
    train_data = [
        {"text": "Refund my order please", "label": "refund", "type": "canonical"},
        {"text": "Give my money back", "label": "refund", "type": "canonical"},
        {"text": "Cancel my subscription", "label": "cancel", "type": "canonical"},
        {"text": "Stop the service", "label": "cancel", "type": "canonical"},
    ]
    test_data = [
        {"text": "I need my refund immediately", "label": "refund", "type": "boundary"},
        {"text": "Cancel my order right away", "label": "cancel", "type": "boundary"},
        {"text": "Don't charge me, refund this", "label": "refund", "type": "hard_negative"},
    ]

    trainer = train_classifier(train_data, framework="sklearn_fast", seed=42)
    eval_res = evaluate_model(trainer["predictor"], test_data)

    assert "macro_f1" in eval_res
    assert "accuracy" in eval_res
    assert "per_class_metrics" in eval_res
    assert "confusion_matrix" in eval_res
    assert "boundary_accuracy" in eval_res
    assert "hard_negative_accuracy" in eval_res
    assert eval_res["total_evaluated"] == 3
    assert len(eval_res["detailed_results"]) == 3
    assert "classes" in eval_res["confusion_matrix"]
    assert "matrix" in eval_res["confusion_matrix"]
