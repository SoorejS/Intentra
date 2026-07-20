"""
Intentra v2 - FastAPI Backend (Productized)
Connects the intent-driven pipeline to the frontend, saves to SQLite, and supports exports.

Run with: uvicorn main:app --reload --port 8000
"""

import json
import os
import sys
import csv
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
from database import engine, get_db
import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intentra API",
    description="Intent-Driven LLM Training Platform",
    version="2.1"
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

        # Step 3: Evaluate quality
        evaluation = full_evaluation(dataset, schema)

        # Step 4: Build summary
        adversarial_count = sum(1 for ex in dataset if ex.get("type") == "adversarial")
        summary = {
            "total_examples": len(dataset),
            "adversarial_examples": adversarial_count,
            "canonical_examples": sum(1 for ex in dataset if ex.get("type") == "canonical"),
            "boundary_examples": sum(1 for ex in dataset if ex.get("type") == "boundary"),
            "overall_quality_score": evaluation["intent_quality"]["overall_score"],
            "ready_for_training": evaluation["ready_for_training"],
            "classes": [c["label"] for c in schema.get("output_classes", [])]
        }
        
        # Step 5: Save to database
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
    """Export a generated dataset as CSV or JSONL."""
    db_gen = db.query(models.Generation).filter(models.Generation.id == gen_id).first()
    if not db_gen:
        raise HTTPException(status_code=404, detail="Generation not found")
        
    dataset = db_gen.dataset
    
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
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'jsonl'")


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
        "overall_quality_score": db_gen.evaluation["intent_quality"]["overall_score"],
        "ready_for_training": db_gen.evaluation["ready_for_training"],
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
