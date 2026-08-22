"""
Unit Tests for Intentra V2.1 Curriculum Scheduler Module.
"""

from core.curriculum_scheduler import (
    CurriculumPolicy,
    determine_curriculum_stage,
    build_curriculum_dataset
)


def test_curriculum_policy_quotas():
    policy = CurriculumPolicy(min_canonical_ratio=0.35, max_hard_negative_ratio=0.20)

    # Stage 1: Anchor Grounding (70% canonical, 30% variation, 0 boundary, 0 hard-neg)
    q1 = policy.calculate_stage_quotas(total_count=100, current_stage=1)
    assert q1["canonical_anchor"] == 70
    assert q1["controlled_variation"] == 30
    assert q1["boundary_case"] == 0
    assert q1["hard_negative"] == 0
    assert sum(q1.values()) == 100

    # Stage 2: Controlled Variation
    q2 = policy.calculate_stage_quotas(total_count=100, current_stage=2)
    assert q2["canonical_anchor"] == 50
    assert q2["boundary_case"] == 15
    assert q2["hard_negative"] == 0
    assert sum(q2.values()) == 100

    # Stage 3: Boundary Disambiguation
    q3 = policy.calculate_stage_quotas(total_count=100, current_stage=3)
    assert q3["canonical_anchor"] >= 35 # Preserves anchor floor
    assert q3["boundary_case"] == 30
    assert q3["hard_negative"] == 10
    assert sum(q3.values()) == 100

    # Stage 4: Contrastive Hardening
    q4 = policy.calculate_stage_quotas(total_count=100, current_stage=4)
    assert q4["canonical_anchor"] >= 35
    assert q4["hard_negative"] <= 20 # Bounded
    assert sum(q4.values()) == 100


def test_determine_curriculum_stage_progression():
    # No history -> Stage 1
    s1, r1 = determine_curriculum_stage([])
    assert s1 == 1

    # Low F1 -> Stage 2
    s2, r2 = determine_curriculum_stage([{"macro_f1": 0.40}], current_f1=0.40)
    assert s2 == 2

    # Medium F1 with confusion -> Stage 3
    s3, r3 = determine_curriculum_stage([{"macro_f1": 0.40}, {"macro_f1": 0.55}], current_f1=0.55, confusion_severity="HIGH")
    assert s3 == 3

    # High F1 -> Stage 4
    s4, r4 = determine_curriculum_stage([{"macro_f1": 0.68}, {"macro_f1": 0.72}], current_f1=0.72, confusion_severity="LOW")
    assert s4 == 4


def test_build_curriculum_dataset_provenance():
    classes = ["refund_request", "cancellation_request"]
    data = build_curriculum_dataset(total_count=20, stage=3, classes=classes, seed=42)

    assert len(data) == 20
    for item in data:
        assert "text" in item
        assert "label" in item
        assert item["label"] in classes
        assert "generation_stage" in item
        assert "archetype" in item
        assert "difficulty" in item
        assert "generation_reason" in item
        assert item["generation_stage"] in [1, 2, 3, 4]
