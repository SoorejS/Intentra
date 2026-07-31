"""
Intentra End-to-End System Debug & Diagnostics Suite
Tests all modules, database models, auth, job manager, evaluator, sanity check, and API endpoints.
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
print("[INTENTRA SYSTEM DIAGNOSTICS & SUITE CHECK]")
print("="*70)

errors = []
warnings = []

# 1. Check Provider Configuration
print("\n[1/7] Testing LLM Provider Detection & Client...")
try:
    from core.llm_client import _detect_provider, _is_real_key, _get_provider_models
    provider = _detect_provider()
    models = _get_provider_models()
    print(f"  + Active Provider: {provider}")
    print(f"  + Model: {models.get(provider) if provider else 'None'}")
    print(f"  + GROQ Key Configured: {_is_real_key(os.environ.get('GROQ_API_KEY'))}")
    print(f"  + OPENROUTER Key Configured: {_is_real_key(os.environ.get('OPENROUTER_API_KEY'))}")
    print(f"  + ANTHROPIC Key Configured: {_is_real_key(os.environ.get('ANTHROPIC_API_KEY'))}")
except Exception as e:
    errors.append(f"LLM Client test failed: {e}")

# 2. Check JSON Extractor Robustness
print("\n[2/7] Testing Robust JSON Extractor...")
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
print("\n[3/7] Testing Auth Module (JWT & Hashing)...")
try:
    from core.auth import hash_password, verify_password, create_token, decode_token
    h = hash_password("testpass123")
    assert verify_password("testpass123", h), "Password verification failed"
    assert not verify_password("wrongpass", h), "Invalid password accepted"
    token = create_token(42, "user@test.com")
    payload = decode_token(token)
    assert payload and payload["user_id"] == 42, "Token decoding failed"
    print("  + Auth hashing & token verification passed!")
except Exception as e:
    errors.append(f"Auth module failed: {e}")

# 4. Check Database Models & Connection
print("\n[4/7] Testing Database Engine & Schema...")
try:
    from database import engine, get_db, auto_migrate_sqlite
    import models
    models.Base.metadata.create_all(bind=engine)
    auto_migrate_sqlite()
    db = next(get_db())
    
    # Query test
    gens_count = db.query(models.Generation).count()
    users_count = db.query(models.User).count()
    projects_count = db.query(models.Project).count()
    print(f"  + Database connection OK! (Generations: {gens_count}, Users: {users_count}, Projects: {projects_count})")
except Exception as e:
    errors.append(f"Database check failed: {e}")

# 5. Check Sanity Check & Label Normalization
print("\n[5/7] Testing Sanity Check & Label Normalization...")
try:
    from core.sanity_check import run_sanity_check
    mock_schema = {
        "output_classes": [
            {"label": "Complaint", "description": "Dissatisfaction"},
            {"label": "Praise", "description": "Positive feedback"}
        ]
    }
    mock_dataset = [
        {"text": "I am upset with this product.", "label": "Employee Complaint", "type": "canonical"},
        {"text": "I am upset with this product.", "label": "Complaint", "type": "canonical"}, # duplicate text
        {"text": "Wonderful service!", "label": "Praise", "type": "canonical"}
    ]
    result = run_sanity_check(mock_dataset, mock_schema)
    clean = result["clean_dataset"]
    report = result["report"]
    print(f"  + Original: {report['original_count']} -> Final: {report['final_count']} (Dups removed: {report['duplicates_removed']}, Labels normalized: {report['invalid_labels_removed']})")
    assert len(clean) == 2, f"Expected 2 clean items, got {len(clean)}"
    assert clean[0]["label"] == "Complaint", "Label normalization failed"
    print("  + Label normalization & Jaccard deduplication passed!")
except Exception as e:
    errors.append(f"Sanity check failed: {e}")

# 6. Check Quality Evaluator
print("\n[6/7] Testing Evaluator Module...")
try:
    from core.evaluator import full_evaluation
    eval_res = full_evaluation(clean, mock_schema)
    assert "intent_quality" in eval_res, "Missing intent_quality"
    assert "structural_metrics" in eval_res, "Missing structural_metrics"
    assert eval_res["intent_quality"]["overall_score"] > 0, "Invalid overall score"
    print(f"  + Overall Quality Score: {eval_res['intent_quality']['overall_score']}/10")
    print(f"  + Ready for Training: {eval_res['ready_for_training']}")
    print("  + Evaluator passed!")
except Exception as e:
    errors.append(f"Evaluator failed: {e}")

# 7. Check FastAPI App & Route Definition
print("\n[7/7] Testing FastAPI Route Registrations...")
try:
    from main import app
    route_paths = [r.path for r in app.routes]
    expected_routes = [
        "/api/health", "/api/debug", "/api/provider", "/api/benchmark",
        "/api/generate", "/api/refine", "/api/jobs/{job_id}/stream",
        "/api/auth/signup", "/api/auth/login", "/api/auth/me",
        "/api/projects", "/api/analytics", "/api/history"
    ]
    for er in expected_routes:
        assert er in route_paths, f"Missing route: {er}"
    print(f"  + All {len(expected_routes)} API endpoints registered cleanly!")
except Exception as e:
    errors.append(f"FastAPI route check failed: {e}")

print("\n" + "="*70)
if errors:
    print(f"FAILED WITH {len(errors)} ERROR(S):")
    for err in errors:
        print(f"   * {err}")
else:
    print("SUCCESS: ALL SYSTEM DIAGNOSTICS PASSED WITH ZERO ERRORS!")
print("="*70)
