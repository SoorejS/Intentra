"""
Unit Tests for Intentra V2 Targeted Data Generator & Quality Filter.
"""

from core.targeted_generator import generate_targeted_data
from core.quality_filter import filter_candidate_examples


def test_targeted_generation_structure():
    schema = {
        "output_classes": [
            {"label": "refund_request"},
            {"label": "cancel_order"}
        ]
    }
    diagnostics = {
        "primary_focus_pair": ("refund_request", "cancel_order"),
        "primary_focus_weak_class": "cancel_order",
        "target_problem_summary": "Confusion between refund_request and cancel_order"
    }

    results = generate_targeted_data(schema, diagnostics, count=10)
    assert len(results) >= 10
    for item in results:
        assert "text" in item
        assert item["label"] in ["refund_request", "cancel_order"]
        assert item["type"] in ["boundary", "hard_negative", "canonical"]
        assert "generation_reason" in item


def test_quality_filter_deduplication():
    existing = [{"text": "I want a full refund for this damaged product.", "label": "refund_request"}]
    candidates = [
        {"text": "I want a full refund for this damaged product.", "label": "refund_request"}, # Exact dup
        {"text": "Please refund my damaged product completely.", "label": "refund_request"}, # Different
        {"text": "Hi", "label": "refund_request"}, # Too short
        {"text": "Cancel my order please.", "label": "unknown_class"} # Invalid label
    ]
    schema = {"output_classes": [{"label": "refund_request"}, {"label": "cancel_order"}]}

    res = filter_candidate_examples(candidates, existing, schema)

    assert len(res["accepted"]) == 1
    assert res["accepted"][0]["text"] == "Please refund my damaged product completely."
    assert len(res["rejected"]) == 3
    assert res["telemetry"]["examples_generated"] == 4
    assert res["telemetry"]["examples_accepted"] == 1
    assert res["telemetry"]["examples_rejected"] == 3
