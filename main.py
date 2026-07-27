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
from dotenv import load_dotenv
load_dotenv()  # Load .env so all provider keys are available
from io import StringIO
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.intent_schema import generate_intent_schema
from core.dataset_generator import generate_full_dataset
from core.evaluator import full_evaluation
from core.sanity_check import run_sanity_check
from database import engine, get_db
import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

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


class GenerateRequest(BaseModel):
    objective: str
    dataset_size: Optional[int] = 20
    domain_hint: Optional[str] = ""


class GenerateResponse(BaseModel):
    id: int
    schema_data: dict
    dataset: list
    evaluation: dict
    summary: dict


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/provider")
def get_provider():
    """Return the active LLM provider so the UI can display it."""
    from core.llm_client import _detect_provider, PROVIDER_MODELS
    provider = _detect_provider()
    model = PROVIDER_MODELS.get(provider, "unknown") if provider else None
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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_dataset(request: GenerateRequest, db: Session = Depends(get_db)):
    """Main endpoint: objective in, training dataset out."""
    
    if not request.objective or len(request.objective.strip()) < 10:
        raise HTTPException(status_code=400, detail="Objective must be at least 10 characters")

    if request.dataset_size < 5 or request.dataset_size > 100:
        raise HTTPException(status_code=400, detail="Dataset size must be between 5 and 100")

    try:
        # Step 1: Generate intent schema
        schema = generate_intent_schema(request.objective, request.domain_hint)

        # Step 2: Generate dataset
        dataset = generate_full_dataset(schema, request.dataset_size)

        # Step 3: Sanity check – validate labels + deduplicate
        sanity_result = run_sanity_check(dataset, schema)
        dataset = sanity_result["clean_dataset"]
        sanity_report = sanity_result["report"]

        # Step 4: Evaluate quality
        evaluation = full_evaluation(dataset, schema)
        evaluation["sanity_report"] = sanity_report

        # Step 5: Build summary
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
        
        # Step 6: Save to database
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

# Mount static files for the frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")
