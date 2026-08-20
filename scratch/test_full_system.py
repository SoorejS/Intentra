"""
Intentra V2 End-to-End System Debug & Diagnostics Suite
Tests all V1 & V2 modules, database models, classifier trainer, evaluator,
error analyzer, targeted generator, optimization flywheel, benchmark suite, and API endpoints.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print("="*70)
print("[INTENTRA V2 FULL SYSTEM DIAGNOSTICS & SUITE CHECK]")
print("="*70)

errors = []
warnings = []

# 1. Check Provider Configuration
print("\n[1/9] Testing LLM Provider Detection & Client...")
try:
    from core.llm_client import _detect_provider, _is_real_key, _get_provider_models
    provider = _detect_provider()
    models = _get_provider_models()
    print(f"  + Active Provider: {provider}")
    print(f"  + Model: {models.get(provider) if provider else 'None'}")
    print(f"  + GROQ Key Configured: {_is_real_key(os.environ.get('GROQ_API_KEY'))}")
except Exception as e:
    errors.append(f"LLM Client test failed: {e}")

# 2. Check JSON Extractor Robustness
print("\n[2/9] Testing Robust JSON Extractor...")
try:
    from core.llm_client import extract_json
    test_cases = [
        ('{"a": 1}', {"a": 1}),
        ('```json\n[{"text": "hi"}]\n```', [{"text": "hi"}]),
        ('Here is the json:\n```\n{"b": 2}\n```\nHope it helps!', {"b": 2}),
        ('Sure! {"c": 3} is your data.', {"c": 3}),
    ]
    for raw, expected in test_cases:
        res = extract_json(raw)
        assert res == expected, f"Expected {expected}, got {res}"
    print("  + JSON extractor passed all edge cases!")
except Exception as e:
    errors.append(f"JSON Extractor failed: {e}")

# 3. Check Auth & Token Verification
print("\n[3/9] Testing Auth Module (JWT & Hashing)...")
try:
    from core.auth import hash_password, verify_password, create_token, decode_token
    h = hash_password("testpass123")
    assert verify_password("testpass123", h), "Password verification failed"
    token = create_token(42, "user@test.com")
    payload = decode_token(token)
    assert payload and payload["user_id"] == 42, "Token decoding failed"
    print("  + Auth hashing & token verification passed!")
except Exception as e:
    errors.append(f"Auth module failed: {e}")

# 4. Check Database Models & Connection
print("\n[4/9] Testing Database Engine & V1/V2 Schema...")
try:
    from database import engine, get_db, auto_migrate_sqlite
    import models
    models.Base.metadata.create_all(bind=engine)
    auto_migrate_sqlite()
    db = next(get_db())
    
    gens_count = db.query(models.Generation).count()
    ver_count = db.query(models.DatasetVersion).count()
    train_count = db.query(models.TrainingRun).count()
    opt_count = db.query(models.OptimizationCycle).count()
    print(f"  + Database connection OK! (Generations: {gens_count}, Versions: {ver_count}, TrainingRuns: {train_count}, OptCycles: {opt_count})")
except Exception as e:
    errors.append(f"Database check failed: {e}")

# 5. Check V2 Classifier Trainer & Evaluation Engine
print("\n[5/9] Testing V2 Classifier Trainer & Evaluation Engine...")
try:
    from core.classifier_trainer import train_classifier
    from core.evaluation_engine import evaluate_model

    mock_train = [
        {"text": "I want a refund for the broken shoes.", "label": "refund", "type": "canonical"},
        {"text": "Please return my money back to my card.", "label": "refund", "type": "canonical"},
        {"text": "Cancel my order before it ships.", "label": "cancel", "type": "canonical"},
        {"text": "Stop my subscription now.", "label": "cancel", "type": "canonical"}
    ]
    mock_test = [
        {"text": "Refund my order please.", "label": "refund", "type": "boundary"},
        {"text": "Cancel my active order.", "label": "cancel", "type": "boundary"},
        {"text": "Don't charge me, refund this.", "label": "refund", "type": "hard_negative"}
    ]

    train_res = train_classifier(mock_train, framework="sklearn_fast", seed=42)
    assert "predictor" in train_res
    print(f"  + Trainer trained in {train_res['training_time_seconds']}s")

    eval_res = evaluate_model(train_res["predictor"], mock_test)
    assert "macro_f1" in eval_res
    assert "boundary_accuracy" in eval_res
    assert "hard_negative_accuracy" in eval_res
    print(f"  + Evaluation: Macro F1 = {eval_res['macro_f1']:.4f}, Boundary Acc = {eval_res['boundary_accuracy']:.4f}")
except Exception as e:
    errors.append(f"Classifier Trainer / Evaluator failed: {e}")

# 6. Check V2 Error Analyzer
print("\n[6/9] Testing V2 Error Analyzer & Boundary Diagnosis...")
try:
    from core.error_analyzer import analyze_errors
    diag = analyze_errors(eval_res)
    assert "weakest_classes" in diag
    assert "target_problem_summary" in diag
    print(f"  + Diagnostic: {diag['target_problem_summary']}")
except Exception as e:
    errors.append(f"Error Analyzer failed: {e}")

# 7. Check V2 Targeted Generator & Quality Filter
print("\n[7/9] Testing V2 Targeted Generator & Quality Filter...")
try:
    from core.targeted_generator import generate_targeted_data
    from core.quality_filter import filter_candidate_examples

    schema = {"output_classes": [{"label": "refund"}, {"label": "cancel"}]}
    targeted = generate_targeted_data(schema, diag, count=10)
    assert len(targeted) >= 10

    filtered = filter_candidate_examples(targeted, mock_train, schema)
    assert len(filtered["accepted"]) > 0
    print(f"  + Generated {len(targeted)} targeted items, accepted {len(filtered['accepted'])} items ({filtered['telemetry']['acceptance_rate']*100:.1f}%)")
except Exception as e:
    errors.append(f"Targeted Generator / Quality Filter failed: {e}")

# 8. Check V2 Closed-Loop Optimization Flywheel
print("\n[8/9] Testing V2 Optimization Flywheel (Base -> Retrain -> Gate)...")
try:
    import uuid
    from core.optimization_engine import run_optimization_cycle

    # Create dummy base version
    v_base = models.DatasetVersion(
        id=str(uuid.uuid4()),
        version_number=1,
        total_examples=len(mock_train),
        generated_by="initial",
        status="promoted",
        dataset_json=json.dumps(mock_train),
        schema_json=json.dumps(schema)
    )
    db.add(v_base)
    db.commit()

    cycle_res = run_optimization_cycle(
        db=db,
        base_version_id=v_base.id,
        val_dataset=mock_test,
        framework="sklearn_fast",
        targeted_count=10,
        seed=42
    )
    assert "cycle_id" in cycle_res
    assert "status" in cycle_res
    print(f"  + Cycle verdict: {cycle_res['status'].upper()} (F1: {cycle_res['metrics_diff']['baseline_f1']:.3f} -> {cycle_res['metrics_diff']['resulting_f1']:.3f})")
except Exception as e:
    errors.append(f"Optimization Flywheel failed: {e}")

# 9. Check FastAPI V1 & V2 Routes Registration
print("\n[9/9] Testing FastAPI Route Registrations...")
try:
    from main import app
    routes = [route.path for route in app.routes]
    expected = [
        "/api/health", "/api/generate", "/api/refine",
        "/api/train", "/api/evaluate", "/api/errors", "/api/errors/analysis",
        "/api/optimize", "/api/datasets/versions", "/api/benchmarks"
    ]
    for exp in expected:
        assert exp in routes, f"Missing route: {exp}"
    print(f"  + All {len(routes)} API endpoints (V1 + V2) registered cleanly!")
except Exception as e:
    errors.append(f"FastAPI route check failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
if errors:
    print(f"FAILED WITH {len(errors)} ERROR(S):")
    for err in errors:
        print(f"   * {err}")
    print("="*70)
    sys.exit(1)
else:
    print("SUCCESS: ALL INTENTRA V1 & V2 SYSTEM DIAGNOSTICS PASSED WITH ZERO ERRORS!")
    print("="*70)
