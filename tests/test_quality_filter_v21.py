"""
Unit Tests for Intentra V2.1 Quality Filter Module.
"""

from core.quality_filter import filter_candidate_examples


def test_quality_filter_cross_class_keyword_detection():
    existing = []
    schema = {
        "output_classes": [
            {"label": "refund_request"},
            {"label": "cancellation_request"}
        ]
    }

    candidates = [
        # Clean canonical anchor
        {"text": "Please return my money back to my account.", "label": "refund_request", "archetype": "canonical_anchor", "type": "canonical"},
        # Contaminated canonical anchor with strong cancellation keywords
        {"text": "Cancel and abort my account renewal and reverse charge.", "label": "refund_request", "archetype": "canonical_anchor", "type": "canonical"},
        # Valid boundary case where mixed keywords are allowed
        {"text": "Can I cancel and request a refund reversal?", "label": "refund_request", "archetype": "boundary_case", "type": "boundary"}
    ]

    res = filter_candidate_examples(candidates, existing, schema, curriculum_stage=1)

    assert len(res["accepted"]) == 2
    assert res["accepted"][0]["text"] == "Please return my money back to my account."
    assert res["accepted"][1]["text"] == "Can I cancel and request a refund reversal?"

    assert len(res["rejected"]) == 1
    assert "Cross-class keyword contamination" in res["rejected"][0]["rejection_reason"]


def test_quality_filter_provenance_enrichment():
    existing = []
    schema = {"output_classes": [{"label": "refund_request"}]}
    candidates = [
        {"text": "I am demanding an immediate refund for order #101.", "label": "refund_request"}
    ]

    res = filter_candidate_examples(candidates, existing, schema, curriculum_stage=2)
    assert len(res["accepted"]) == 1
    item = res["accepted"][0]
    assert item["generation_stage"] == 2
    assert "archetype" in item
    assert "difficulty" in item
    assert "target_class" in item
    assert res["telemetry"]["anchor_coverage_ratio"] == 1.0
