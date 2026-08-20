"""
Unit Tests for Intentra V2 Optimization Engine & Closed-Loop Flywheel.
"""

import json
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
import models
from core.optimization_engine import run_optimization_cycle, evaluate_promotion_gate


def get_test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_promotion_gate_logic():
    # 1. Clear improvement -> Promoted
    base_eval = {"macro_f1": 0.70, "boundary_accuracy": 0.60, "hard_negative_accuracy": 0.55}
    cand_eval = {"macro_f1": 0.76, "boundary_accuracy": 0.72, "hard_negative_accuracy": 0.68}
    diag = {"primary_focus_pair": ("A", "B")}
    is_promoted, rationale, diff = evaluate_promotion_gate(base_eval, cand_eval, diag)
    assert is_promoted is True
    assert diff["delta_f1"] == 0.06

    # 2. Insufficient lift (< +0.010) -> Rejected
    cand_eval_flat = {"macro_f1": 0.704, "boundary_accuracy": 0.60, "hard_negative_accuracy": 0.55}
    is_promoted_flat, rationale_flat, _ = evaluate_promotion_gate(base_eval, cand_eval_flat, diag)
    assert is_promoted_flat is False
    assert "Rejected" in rationale_flat

    # 3. Severe regression on boundary slice -> Rejected
    cand_eval_regr = {"macro_f1": 0.72, "boundary_accuracy": 0.50, "hard_negative_accuracy": 0.55}
    is_promoted_regr, rationale_regr, _ = evaluate_promotion_gate(base_eval, cand_eval_regr, diag)
    assert is_promoted_regr is False
    assert "Boundary" in rationale_regr


def test_full_optimization_cycle_execution():
    db = get_test_db()

    initial_dataset = [
        {"text": "I demand a refund for the defective microwave.", "label": "refund_request", "type": "canonical"},
        {"text": "Please return my purchase money to Visa.", "label": "refund_request", "type": "canonical"},
        {"text": "Cancel my order before delivery.", "label": "cancellation_request", "type": "canonical"},
        {"text": "Stop my subscription renewal immediately.", "label": "cancellation_request", "type": "canonical"},
        {"text": "Can I cancel and get a refund?", "label": "cancellation_request", "type": "boundary"},
        {"text": "I don't want refund, just cancel.", "label": "cancellation_request", "type": "hard_negative"}
    ]
    schema = {
        "output_classes": [
            {"label": "refund_request"},
            {"label": "cancellation_request"}
        ]
    }

    # Create v1
    v1 = models.DatasetVersion(
        id=str(uuid.uuid4()),
        version_number=1,
        total_examples=len(initial_dataset),
        generated_by="initial",
        status="promoted",
        dataset_json=json.dumps(initial_dataset),
        schema_json=json.dumps(schema)
    )
    db.add(v1)
    db.commit()

    val_dataset = [
        {"text": "Please refund my account.", "label": "refund_request", "type": "canonical"},
        {"text": "Cancel my delivery order.", "label": "cancellation_request", "type": "canonical"},
        {"text": "Don't send the box, cancel it.", "label": "cancellation_request", "type": "boundary"}
    ]

    res = run_optimization_cycle(
        db=db,
        base_version_id=v1.id,
        val_dataset=val_dataset,
        framework="sklearn_fast",
        targeted_count=10,
        seed=42
    )

    assert "cycle_id" in res
    assert "status" in res
    assert "metrics_diff" in res
    assert res["candidate_version"]["version_number"] == 2

    # Verify database persistence
    cand_version = db.query(models.DatasetVersion).filter(models.DatasetVersion.version_number == 2).first()
    assert cand_version is not None
    assert cand_version.parent_version_id == v1.id
    assert cand_version.total_examples > v1.total_examples

    opt_cycle = db.query(models.OptimizationCycle).filter(models.OptimizationCycle.id == res["cycle_id"]).first()
    assert opt_cycle is not None
    assert opt_cycle.base_dataset_version_id == v1.id
    assert opt_cycle.resulting_dataset_version_id == cand_version.id
