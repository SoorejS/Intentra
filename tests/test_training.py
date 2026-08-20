"""
Unit Tests for Intentra V2 Classifier Trainer Module.
"""

import pytest
from core.classifier_trainer import train_classifier, SklearnPredictor


def get_mock_dataset():
    return [
        {"text": "I want a refund for my damaged shoes immediately.", "label": "refund_request", "type": "canonical"},
        {"text": "Please return my payment to my card.", "label": "refund_request", "type": "canonical"},
        {"text": "Can I get my funds back if the item is broken?", "label": "refund_request", "type": "boundary"},
        {"text": "Cancel my order right now before it ships.", "label": "cancellation_request", "type": "canonical"},
        {"text": "Please stop my active subscription.", "label": "cancellation_request", "type": "canonical"},
        {"text": "I do not want this order anymore, cancel it.", "label": "cancellation_request", "type": "boundary"}
    ]


def test_train_classifier_empty_fails():
    with pytest.raises(ValueError, match="empty dataset"):
        train_classifier([])


def test_train_classifier_single_class_fails():
    single_class_data = [{"text": "Hello world", "label": "single_class"}]
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        train_classifier(single_class_data)


def test_train_classifier_sklearn_fast_reproducibility():
    data = get_mock_dataset()
    res1 = train_classifier(data, framework="sklearn_fast", seed=42)
    res2 = train_classifier(data, framework="sklearn_fast", seed=42)

    assert isinstance(res1["predictor"], SklearnPredictor)
    assert res1["classes"] == ["cancellation_request", "refund_request"]
    assert res1["training_time_seconds"] >= 0.0

    preds1, confs1 = res1["predictor"].predict(["I want my money back", "Cancel my shipment"])
    preds2, confs2 = res2["predictor"].predict(["I want my money back", "Cancel my shipment"])

    assert preds1 == preds2
    assert confs1 == confs2
    assert preds1[0] == "refund_request"
    assert preds1[1] == "cancellation_request"
