"""
Intentra v3 - FastAPI Backend (Enhanced)
- Multi-provider LLM: Groq, OpenRouter, Anthropic, Local
- Benchmark proof-point endpoint
- OpenAI fine-tuning format export
- Dataset sanity check (dedup + label validation)
- Colab notebook download

Run with: uvicorn main:app --reload --port 8000
"""

import json
import os
import sys
import csv
import textwrap
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load .env so all provider keys are available
from io import StringIO
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.intent_schema import generate_intent_schema
from core.dataset_generator import generate_full_dataset
from core.evaluator import full_evaluation
from core.sanity_check import run_sanity_check
from core.job_manager import job_manager
from database import engine, get_db, auto_migrate_sqlite
import models

# Create database tables & run auto migrations
models.Base.metadata.create_all(bind=engine)
auto_migrate_sqlite()

app = FastAPI(
    title="Intentra API",
    description="Intent-Driven LLM Training Platform",
    version="3.0"
)

# Allow frontend to call this API (useful during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def render_keep_alive_loop():
    """Background task to self-ping every 10 minutes to prevent Render free-tier sleep."""
    import urllib.request
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "https://intentra-jvd1.onrender.com").rstrip("/") + "/api/health"
    print(f"[Keep-Alive] Self-ping keep-alive service initialized for: {render_url}")
    while True:
        await asyncio.sleep(600) # Ping every 10 minutes (600s)
        try:
            req = urllib.request.Request(render_url, headers={"User-Agent": "Intentra-KeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                print(f"[Keep-Alive] Self-ping OK: {response.status}")
        except Exception as e:
            print(f"[Keep-Alive] Self-ping ping result: {e}")


@app.on_event("startup")
async def start_keep_alive():
    asyncio.create_task(render_keep_alive_loop())


class GenerateRequest(BaseModel):
    objective: str
    dataset_size: Optional[int] = 20
    domain_hint: Optional[str] = ""
    target_language: Optional[str] = "English"
    is_multilabel: Optional[bool] = False
    custom_classes: Optional[List[str]] = None


class RefineRequest(BaseModel):
    generation_id: int
    instruction: str
    additional_count: Optional[int] = 10
    target_language: Optional[str] = "English"


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ProjectRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class GenerateResponse(BaseModel):
    id: int
    schema_data: dict
    dataset: list
    evaluation: dict
    summary: dict


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/debug")
def debug_config():
    """Debug endpoint - shows provider config and tests connectivity."""
    from core.llm_client import _detect_provider, _is_real_key, _get_provider_models
    provider = _detect_provider()
    models = _get_provider_models()
    
    keys_status = {
        "GROQ_API_KEY": _is_real_key(os.environ.get("GROQ_API_KEY")),
        "OPENROUTER_API_KEY": _is_real_key(os.environ.get("OPENROUTER_API_KEY")),
        "ANTHROPIC_API_KEY": _is_real_key(os.environ.get("ANTHROPIC_API_KEY")),
        "LOCAL_API_KEY": _is_real_key(os.environ.get("LOCAL_API_KEY")),
        "LOCAL_BASE_URL": bool(os.environ.get("LOCAL_BASE_URL")),
    }
    
    # Quick LLM test
    test_result = None
    try:
        from core.llm_client import call_llm
        response = call_llm("Reply with exactly: INTENTRA_OK", max_tokens=20, temperature=0.0)
        test_result = {"success": True, "response": response[:100]}
    except Exception as e:
        test_result = {"success": False, "error": str(e)}
    
    return {
        "active_provider": provider,
        "model": models.get(provider) if provider else None,
        "keys_configured": keys_status,
        "llm_test": test_result,
        "env_loaded": bool(os.environ.get("LOCAL_API_KEY")),
    }


@app.get("/api/provider")
def get_provider():
    """Return the active LLM provider so the UI can display it."""
    from core.llm_client import _detect_provider, _get_provider_models
    provider = _detect_provider()
    models = _get_provider_models()
    model = models.get(provider, "unknown") if provider else None
    return {
        "provider": provider or "none",
        "model": model,
        "configured": provider is not None
    }


@app.get("/api/benchmark")
def get_benchmark():
    """Return the empirical benchmark results for display in the UI."""
    return {
        "intentra": {
            "mean_f1": 0.6667,
            "std_f1": 0.0000,
            "seeds": [0.6667, 0.6667, 0.6667],
            "label": "Intentra (Adversarial Dataset)"
        },
        "naive": {
            "mean_f1": 0.6510,
            "std_f1": 0.0334,
            "seeds": [0.6176, 0.6863, 0.6490],
            "label": "Naive Baseline"
        },
        "improvement_pct": round(((0.6667 - 0.6510) / 0.6510) * 100, 1),
        "std_reduction_pct": round((1 - (0.0000 / 0.0334)) * 100, 1),
        "model": "distilbert-base-uncased",
        "test_set": "30 hand-written adversarial examples (15 per class)",
        "seeds_tested": 3,
        "verdict": "Intentra datasets produce higher and more stable fine-tuning performance"
    }


async def run_async_generation_task(job_id: str, objective: str, dataset_size: int, domain_hint: str,
                                     target_language: str = "English", is_multilabel: bool = False,
                                     custom_classes: list = None):
    """Background task executing the generation pipeline in worker threads and emitting SSE progress events."""
    loop = asyncio.get_running_loop()
    try:
        # Step 1: Generate schema in thread worker
        await job_manager.emit_event(job_id, "status", {"step": 1, "message": "Generating intent schema...", "progress": 15})
        schema = await asyncio.to_thread(generate_intent_schema, objective, domain_hint,
                                          target_language=target_language, is_multilabel=is_multilabel,
                                          custom_classes=custom_classes)
        await job_manager.emit_event(job_id, "schema", {"schema_data": schema, "progress": 30})

        # Step 2: Generate dataset with batch streaming in thread worker
        lang_note = f" in {target_language}" if target_language != "English" else ""
        await job_manager.emit_event(job_id, "status", {"step": 2, "message": f"Synthesizing dataset examples{lang_note}...", "progress": 35})
        
        def on_batch(batch_num, total_batches, batch_examples):
            pct = 35 + int((batch_num / total_batches) * 45)
            asyncio.run_coroutine_threadsafe(
                job_manager.emit_event(job_id, "batch", {
                    "batch_index": batch_num,
                    "total_batches": total_batches,
                    "examples": batch_examples,
                    "progress": pct
                }),
                loop
            )

        dataset = await asyncio.to_thread(generate_full_dataset, schema, dataset_size, on_batch,
                                            target_language=target_language, is_multilabel=is_multilabel)

        # VALIDATION: fail the job if we got 0 examples
        if not dataset or len(dataset) == 0:
            raise RuntimeError(
                "Dataset generation produced 0 examples. "
                "Check that your LLM provider is configured correctly and reachable."
            )

        # Step 3: Sanity check & deduplication
        await job_manager.emit_event(job_id, "status", {"step": 3, "message": "Deduplicating and running sanity checks...", "progress": 85})
        sanity_result = await asyncio.to_thread(run_sanity_check, dataset, schema)
        clean_dataset = sanity_result["clean_dataset"]
        sanity_report = sanity_result["report"]
        await job_manager.emit_event(job_id, "sanity", {"report": sanity_report, "progress": 90})

        # Step 4: Evaluate quality
        evaluation = await asyncio.to_thread(full_evaluation, clean_dataset, schema)
        evaluation["sanity_report"] = sanity_report

        adversarial_count = sum(1 for ex in clean_dataset if ex.get("type") == "adversarial")
        summary = {
            "total_examples": len(clean_dataset),
            "adversarial_examples": adversarial_count,
            "canonical_examples": sum(1 for ex in clean_dataset if ex.get("type") == "canonical"),
            "boundary_examples": sum(1 for ex in clean_dataset if ex.get("type") == "boundary"),
            "overall_quality_score": evaluation["intent_quality"]["overall_score"],
            "ready_for_training": evaluation["ready_for_training"],
            "classes": [c["label"] for c in schema.get("output_classes", [])],
            "duplicates_removed": sanity_report["duplicates_removed"],
            "invalid_labels_removed": sanity_report["invalid_labels_removed"]
        }

        # Step 5: Save to SQLite database
        def save_to_db():
            db = next(get_db())
            try:
                db_generation = models.Generation(
                    objective=objective,
                    domain_hint=domain_hint,
                    schema_json=json.dumps(schema),
                    dataset_json=json.dumps(clean_dataset),
                    evaluation_json=json.dumps(evaluation)
                )
                db.add(db_generation)
                db.commit()
                db.refresh(db_generation)
                return db_generation.id
            finally:
                db.close()

        gen_id = await asyncio.to_thread(save_to_db)

        # Emit completion event with full payload
        await job_manager.emit_event(job_id, "complete", {
            "id": gen_id,
            "schema_data": schema,
            "dataset": clean_dataset,
            "evaluation": evaluation,
            "summary": summary
        })

    except Exception as e:
        import traceback
        print(f"[main] Async job {job_id} FAILED: {e}")
        traceback.print_exc()
        await job_manager.emit_event(job_id, "job_error", {"detail": str(e)})


@app.post("/api/generate")
async def generate_dataset_async(request: GenerateRequest):
    """Async endpoint: returns job_id immediately (<50ms), generation runs in background."""
    if not request.objective or len(request.objective.strip()) < 10:
        raise HTTPException(status_code=400, detail="Objective must be at least 10 characters")

    if request.dataset_size < 5 or request.dataset_size > 1000:
        raise HTTPException(status_code=400, detail="Dataset size must be between 5 and 1000")

    job_id = job_manager.create_job()
    asyncio.create_task(run_async_generation_task(
        job_id, request.objective, request.dataset_size, request.domain_hint or "",
        target_language=request.target_language or "English",
        is_multilabel=request.is_multilabel or False,
        custom_classes=request.custom_classes
    ))

    return {"job_id": job_id, "status": "processing"}


@app.post("/api/refine")
async def refine_dataset(request: RefineRequest, db: Session = Depends(get_db)):
    """Append additional examples to an existing generation based on a refinement instruction."""
    gen = db.query(models.Generation).filter(models.Generation.id == request.generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    schema = json.loads(gen.schema_json)
    existing_dataset = json.loads(gen.dataset_json)

    # Generate additional examples with the refinement instruction baked in
    from core.dataset_generator import generate_full_dataset
    new_examples = generate_full_dataset(
        schema, request.additional_count, None,
        target_language=request.target_language or "English",
        refinement_instruction=request.instruction
    )

    # Merge and re-run sanity check
    merged = existing_dataset + new_examples
    sanity_result = run_sanity_check(merged, schema)
    clean = sanity_result["clean_dataset"]
    evaluation = full_evaluation(clean, schema)
    evaluation["sanity_report"] = sanity_result["report"]

    # Update database record
    gen.dataset_json = json.dumps(clean)
    gen.evaluation_json = json.dumps(evaluation)
    db.commit()

    return {
        "id": gen.id,
        "new_examples_added": len(new_examples),
        "total_examples": len(clean),
        "dataset": clean,
        "evaluation": evaluation
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(job_id: str):
    """Stream SSE real-time events for a generation job."""
    return StreamingResponse(
        job_manager.stream_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll job status and result."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"]
    }


@app.post("/api/generate_sync", response_model=GenerateResponse)
async def generate_dataset_sync(request: GenerateRequest, db: Session = Depends(get_db)):
    """Synchronous fallback endpoint for backward compatibility."""
    if not request.objective or len(request.objective.strip()) < 10:
        raise HTTPException(status_code=400, detail="Objective must be at least 10 characters")

    if request.dataset_size < 5 or request.dataset_size > 100:
        raise HTTPException(status_code=400, detail="Dataset size must be between 5 and 100")

    try:
        schema = generate_intent_schema(request.objective, request.domain_hint)
        dataset = generate_full_dataset(schema, request.dataset_size)
        sanity_result = run_sanity_check(dataset, schema)
        dataset = sanity_result["clean_dataset"]
        sanity_report = sanity_result["report"]

        evaluation = full_evaluation(dataset, schema)
        evaluation["sanity_report"] = sanity_report

        adversarial_count = sum(1 for ex in dataset if ex.get("type") == "adversarial")
        summary = {
            "total_examples": len(dataset),
            "adversarial_examples": adversarial_count,
            "canonical_examples": sum(1 for ex in dataset if ex.get("type") == "canonical"),
            "boundary_examples": sum(1 for ex in dataset if ex.get("type") == "boundary"),
            "overall_quality_score": evaluation["intent_quality"]["overall_score"],
            "ready_for_training": evaluation["ready_for_training"],
            "classes": [c["label"] for c in schema.get("output_classes", [])],
            "duplicates_removed": sanity_report["duplicates_removed"],
            "invalid_labels_removed": sanity_report["invalid_labels_removed"]
        }
        
        db_generation = models.Generation(
            objective=request.objective,
            domain_hint=request.domain_hint,
            schema_json=json.dumps(schema),
            dataset_json=json.dumps(dataset),
            evaluation_json=json.dumps(evaluation)
        )
        db.add(db_generation)
        db.commit()
        db.refresh(db_generation)

        return GenerateResponse(
            id=db_generation.id,
            schema_data=schema,
            dataset=dataset,
            evaluation=evaluation,
            summary=summary
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    """Fetch past generations."""
    generations = db.query(models.Generation).order_by(models.Generation.created_at.desc()).limit(20).all()
    results = []
    for gen in generations:
        results.append({
            "id": gen.id,
            "objective": gen.objective,
            "created_at": gen.created_at,
            "total_examples": len(gen.dataset)
        })
    return {"history": results}


@app.get("/api/export/{gen_id}")
def export_dataset(gen_id: int, format: str = "jsonl", db: Session = Depends(get_db)):
    """Export a generated dataset as CSV, JSONL, or OpenAI fine-tuning format."""
    db_gen = db.query(models.Generation).filter(models.Generation.id == gen_id).first()
    if not db_gen:
        raise HTTPException(status_code=404, detail="Generation not found")
        
    dataset = db_gen.dataset
    schema = db_gen.schema or {}
    
    if format == "csv":
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(["text", "label", "type", "difficulty"])
        for row in dataset:
            cw.writerow([row.get("text", ""), row.get("label", ""), row.get("type", ""), row.get("difficulty", "")])
        
        output = si.getvalue()
        return Response(
            content=output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=intentra_dataset_{gen_id}.csv"}
        )
        
    elif format == "jsonl":
        output = "\n".join([json.dumps(row) for row in dataset])
        return Response(
            content=output,
            media_type="application/jsonl",
            headers={"Content-Disposition": f"attachment; filename=intentra_dataset_{gen_id}.jsonl"}
        )

    elif format == "openai":
        # OpenAI fine-tuning chat format
        lines = []
        for row in dataset:
            entry = {
                "messages": [
                    {"role": "system", "content": f"You are a classifier. Classify the input text into one of: {', '.join([c.get('label','') for c in schema.get('output_classes',[])])}"},
                    {"role": "user", "content": row.get("text", "")},
                    {"role": "assistant", "content": row.get("label", "")}
                ]
            }
            lines.append(json.dumps(entry))
        output = "\n".join(lines)
        return Response(
            content=output,
            media_type="application/jsonl",
            headers={"Content-Disposition": f"attachment; filename=intentra_openai_{gen_id}.jsonl"}
        )
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv', 'jsonl', or 'openai'")


@app.get("/api/export/{gen_id}/notebook")
def export_colab_notebook(gen_id: int, db: Session = Depends(get_db)):
    """Generate and download a Colab-ready Jupyter notebook for fine-tuning."""
    db_gen = db.query(models.Generation).filter(models.Generation.id == gen_id).first()
    if not db_gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    schema = db_gen.schema or {}
    classes = [c.get("label", "") for c in schema.get("output_classes", [])]
    objective = db_gen.objective

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"name": f"Intentra Fine-Tuning - {objective[:40]}", "provenance": []}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Intentra Fine-Tuning Notebook\n\n**Objective:** {objective}\n\n**Classes:** {', '.join(classes)}\n\nGenerated by [Intentra](https://intentra-jvd1.onrender.com) — Intent-Driven LLM Training Platform."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip install transformers datasets scikit-learn torch -q"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json, requests\n",
                    "from datasets import Dataset\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
                    "from sklearn.metrics import f1_score\n",
                    "import numpy as np\n",
                    "\n",
                    f"# Download dataset from Intentra\n",
                    f"response = requests.get('https://intentra-jvd1.onrender.com/api/export/{gen_id}?format=jsonl')\n",
                    "data = [json.loads(line) for line in response.text.strip().split('\\n')]\n",
                    "\n",
                    f"LABEL2ID = {{{', '.join([f'\"'+c+'\": '+str(i) for i,c in enumerate(classes)])}}}\n",
                    f"ID2LABEL = {{{', '.join([str(i)+': \"'+c+'\"' for i,c in enumerate(classes)])}}}\n",
                    "\n",
                    "for ex in data:\n",
                    "    ex['label_id'] = LABEL2ID.get(ex.get('label',''), 0)\n",
                    "\n",
                    "MODEL_NAME = 'distilbert-base-uncased'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                    "\n",
                    "def tokenize(batch):\n",
                    "    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)\n",
                    "\n",
                    "dataset = Dataset.from_list(data).rename_column('label_id', 'labels')\n",
                    "dataset = dataset.map(tokenize, batched=True)\n",
                    "dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])\n",
                    "split = dataset.train_test_split(test_size=0.2, seed=42)\n",
                    "\n",
                    f"model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels={len(classes)}, id2label=ID2LABEL, label2id=LABEL2ID)\n",
                    "\n",
                    "def compute_metrics(p):\n",
                    "    preds = np.argmax(p.predictions, axis=1)\n",
                    "    return {'f1': f1_score(p.label_ids, preds, average='macro')}\n",
                    "\n",
                    "args = TrainingArguments(\n",
                    "    output_dir='./intentra_model',\n",
                    "    num_train_epochs=5,\n",
                    "    per_device_train_batch_size=8,\n",
                    "    evaluation_strategy='epoch',\n",
                    "    save_strategy='epoch',\n",
                    "    load_best_model_at_end=True,\n",
                    "    metric_for_best_model='f1',\n",
                    "    report_to='none'\n",
                    ")\n",
                    "\n",
                    "trainer = Trainer(model=model, args=args,\n",
                    "                  train_dataset=split['train'], eval_dataset=split['test'],\n",
                    "                  compute_metrics=compute_metrics)\n",
                    "\n",
                    "trainer.train()\n",
                    "model.save_pretrained('./intentra_model')\n",
                    "print('Model ready for deployment!')"
                ]
            }
        ]
    }

    return Response(
        content=json.dumps(notebook, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=intentra_colab_{gen_id}.ipynb"}
    )


@app.get("/api/generation/{gen_id}")
def get_generation(gen_id: int, db: Session = Depends(get_db)):
    """Fetch full details of a specific generation."""
    db_gen = db.query(models.Generation).filter(models.Generation.id == gen_id).first()
    if not db_gen:
        raise HTTPException(status_code=404, detail="Generation not found")
        
    adversarial_count = sum(1 for ex in db_gen.dataset if ex.get("type") == "adversarial")
    summary = {
        "total_examples": len(db_gen.dataset),
        "adversarial_examples": adversarial_count,
        "canonical_examples": sum(1 for ex in db_gen.dataset if ex.get("type") == "canonical"),
        "boundary_examples": sum(1 for ex in db_gen.dataset if ex.get("type") == "boundary"),
        "overall_quality_score": db_gen.evaluation.get("intent_quality", {}).get("overall_score", 0),
        "ready_for_training": db_gen.evaluation.get("ready_for_training", False),
        "classes": [c["label"] for c in db_gen.schema.get("output_classes", [])]
    }

    return {
        "id": db_gen.id,
        "schema_data": db_gen.schema,
        "dataset": db_gen.dataset,
        "evaluation": db_gen.evaluation,
        "summary": summary
    }


@app.get("/api/examples")
def get_example_objectives():
    """Return preset example objectives for the frontend."""
    return {
        "examples": [
            {
                "label": "Detect urgent customer complaints",
                "objective": "Detect urgent customer complaints that need immediate escalation",
                "domain_hint": "B2B enterprise customer support"
            },
            {
                "label": "Identify logical fallacies disguised as reasoning",
                "objective": "Identify arguments using logical fallacies disguised as rational reasoning",
                "domain_hint": "debate, political discourse, online argumentation"
            },
            {
                "label": "Classify employee feedback by true intent",
                "objective": "Classify employee feedback by genuine intent behind stated words",
                "domain_hint": "HR systems, employee engagement, workplace communication"
            }
        ]
    }

# ─── Auth Endpoints ─────────────────────────────────────────────────────────
from core.auth import hash_password, verify_password, create_token, decode_token
from fastapi import Header

def get_current_user_optional(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    payload = decode_token(token)
    if not payload:
        return None
    return db.query(models.User).filter(models.User.id == payload["user_id"]).first()

@app.post("/api/auth/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == req.email.lower().strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = models.User(
        email=req.email.lower().strip(),
        hashed_password=hash_password(req.password),
        full_name=req.full_name or ""
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "is_pro": user.is_pro}}

@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == req.email.lower().strip()).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    token = create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "is_pro": user.is_pro}}

@app.get("/api/auth/me")
def get_me(user = Depends(get_current_user_optional)):
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "is_pro": user.is_pro}}

# ─── Projects Endpoints ──────────────────────────────────────────────────────
@app.get("/api/projects")
def get_projects(db: Session = Depends(get_db), user = Depends(get_current_user_optional)):
    query = db.query(models.Project)
    if user:
        query = query.filter((models.Project.user_id == user.id) | (models.Project.user_id == None))
    projects = query.order_by(models.Project.created_at.desc()).all()
    return {"projects": [{"id": p.id, "name": p.name, "description": p.description, "created_at": p.created_at.isoformat()} for p in projects]}

@app.post("/api/projects")
def create_project(req: ProjectRequest, db: Session = Depends(get_db), user = Depends(get_current_user_optional)):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Project name required")
    proj = models.Project(name=req.name.strip(), description=req.description or "", user_id=user.id if user else None)
    db.add(proj)
    db.commit()
    db.refresh(proj)
    return {"id": proj.id, "name": proj.name, "description": proj.description}

# ─── Analytics Endpoint ─────────────────────────────────────────────────────
@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db), user = Depends(get_current_user_optional)):
    total_generations = db.query(models.Generation).count()
    all_gens = db.query(models.Generation).all()
    total_examples = sum(len(g.dataset) for g in all_gens if g.dataset_json)
    
    # Estimate tokens: ~150 tokens per example generated
    estimated_tokens = total_examples * 150
    # Groq Llama 3.1 8B cost: ~$0.05 per 1M tokens -> virtually free
    estimated_cost_usd = round((estimated_tokens / 1_000_000) * 0.05, 4)

    return {
        "total_generations": total_generations,
        "total_examples": total_examples,
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": f"${estimated_cost_usd:.4f}",
        "active_user": user.email if user else "Guest Developer"
    }

# Mount static files for the frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")
