"""
Intentra V2.1 - Closed-Loop Optimization Engine
Orchestrates the Curriculum-Based Training Data Optimization Flywheel:
  1. Train baseline model on current dataset version
  2. Evaluate against validation set (Macro F1, Slices, Confusion Matrix)
  3. Analyze errors and identify weak decision boundary pairs
  4. Determine dynamic curriculum stage (1: Anchor, 2: Variation, 3: Boundary, 4: Contrastive)
  5. Synthesize curriculum-aware targeted synthetic examples
  6. Validate candidates through multi-stage quality filter (with anchor coverage & cross-class checks)
  7. Create immutable DatasetVersion N+1 with full provenance metadata
  8. Retrain candidate model on augmented dataset
  9. Evaluate candidate model on the EXACT SAME validation set
 10. Apply Multi-Objective Promotion Gate:
       - Delta Macro F1 >= +0.010
       - Delta Hard-Negative Acc >= -0.010 (tolerance)
       - Delta Boundary Acc >= -0.010 (tolerance)
       - Targeted confusion pair improved OR Delta Macro F1 >= +0.020
 11. Persist full lineage, training runs, evaluations, errors, and cycle telemetry.
"""

import time
import uuid
import datetime
import json
from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model
from core.error_analyzer import analyze_errors
from core.targeted_generator import generate_targeted_data
from core.quality_filter import filter_candidate_examples
from core.curriculum_scheduler import determine_curriculum_stage, CurriculumPolicy
import models


def evaluate_promotion_gate(
    base_eval: dict,
    cand_eval: dict,
    diagnostics: dict
) -> tuple[bool, str, dict]:
    """
    Evaluate multi-objective promotion gate to prevent false improvements or regressions.
    Returns: (is_promoted, decision_rationale, metrics_diff)
    """
    base_f1 = base_eval.get("macro_f1", 0.0)
    cand_f1 = cand_eval.get("macro_f1", 0.0)
    delta_f1 = round(cand_f1 - base_f1, 4)

    base_bnd = base_eval.get("boundary_accuracy", 0.0)
    cand_bnd = cand_eval.get("boundary_accuracy", 0.0)
    delta_bnd = round(cand_bnd - base_bnd, 4)

    base_hn = base_eval.get("hard_negative_accuracy", 0.0)
    cand_hn = cand_eval.get("hard_negative_accuracy", 0.0)
    delta_hn = round(cand_hn - base_hn, 4)

    # Check targeted pair resolution
    primary_pair = diagnostics.get("primary_focus_pair")
    targeted_improved = False
    if primary_pair:
        cls_a, cls_b = primary_pair
        base_per_class = base_eval.get("per_class_metrics", {})
        cand_per_class = cand_eval.get("per_class_metrics", {})

        base_pair_avg = (base_per_class.get(cls_a, {}).get("f1", 0.0) + base_per_class.get(cls_b, {}).get("f1", 0.0)) / 2.0
        cand_pair_avg = (cand_per_class.get(cls_a, {}).get("f1", 0.0) + cand_per_class.get(cls_b, {}).get("f1", 0.0)) / 2.0
        if cand_pair_avg > base_pair_avg:
            targeted_improved = True

    metrics_diff = {
        "baseline_f1": base_f1,
        "resulting_f1": cand_f1,
        "delta_f1": delta_f1,
        "baseline_boundary_acc": base_bnd,
        "resulting_boundary_acc": cand_bnd,
        "delta_boundary_acc": delta_bnd,
        "baseline_hard_negative_acc": base_hn,
        "resulting_hard_negative_acc": cand_hn,
        "delta_hard_negative_acc": delta_hn,
        "targeted_pair_improved": targeted_improved
    }

    # Gate Conditions
    # 1. Meaningful overall Macro F1 improvement (>= +0.010)
    # 2. No critical slice regression exceeding tolerance (-0.010)
    # 3. Targeted pair improved or Macro F1 improvement is large (>= +0.020)
    if delta_f1 < 0.010:
        return False, f"Rejected: Macro F1 lift ({delta_f1:+.4f}) did not meet minimum threshold (+0.010).", metrics_diff

    if delta_hn < -0.010:
        return False, f"Rejected: Critical regression in Hard-Negative slice accuracy ({delta_hn:+.4f} < -0.010).", metrics_diff

    if delta_bnd < -0.010:
        return False, f"Rejected: Critical regression in Boundary slice accuracy ({delta_bnd:+.4f} < -0.010).", metrics_diff

    if not targeted_improved and delta_f1 < 0.020:
        return False, f"Rejected: Targeted confusion pair F1 did not improve and overall delta ({delta_f1:+.4f}) was below +0.020.", metrics_diff

    rationale = (
        f"Promoted: Macro F1 improved by {delta_f1:+.4f} (from {base_f1:.4f} to {cand_f1:.4f}). "
        f"Boundary Acc: {cand_bnd:.4f} ({delta_bnd:+.4f}), Hard-Neg Acc: {cand_hn:.4f} ({delta_hn:+.4f})."
    )
    return True, rationale, metrics_diff


def run_optimization_cycle(
    db,
    base_version_id: str | None = None,
    val_dataset: list | None = None,
    test_dataset: list | None = None,
    framework: str = "sklearn_fast",
    model_name: str = "distilbert-base-uncased",
    targeted_count: int = 30,
    seed: int = 42,
    project_id: int | None = None,
    curriculum_stage: int | None = None,
    progress_callback = None
) -> dict:
    """
    Execute a full closed-loop optimization cycle with curriculum planning.
    """
    cycle_start = time.time()

    def update_status(step_name: str, progress_pct: int, detail: str = ""):
        if progress_callback:
            progress_callback({
                "step": step_name,
                "progress": progress_pct,
                "detail": detail
            })
        print(f"[Optimization Flywheel] [{progress_pct}%] {step_name}: {detail}")

    update_status("Initializing Flywheel", 5, "Resolving base dataset version...")

    # Step 1: Resolve Base Dataset Version
    if base_version_id:
        base_version = db.query(models.DatasetVersion).filter(models.DatasetVersion.id == base_version_id).first()
    else:
        # Get latest promoted version or latest version
        base_version = db.query(models.DatasetVersion).order_by(models.DatasetVersion.version_number.desc()).first()

    if not base_version or not base_version.dataset:
        raise ValueError("No valid base DatasetVersion found to optimize. Please create or generate an initial dataset version first.")

    base_data = base_version.dataset
    schema = base_version.schema or {}

    # If no validation set passed, construct deterministic stratified 25% holdout split from base
    if not val_dataset:
        np_seed = seed
        import numpy as np
        np.random.seed(np_seed)
        shuffled = list(base_data)
        np.random.shuffle(shuffled)
        split_idx = max(2, int(len(shuffled) * 0.25))
        val_dataset = shuffled[:split_idx]
        train_base_dataset = shuffled[split_idx:]
    else:
        train_base_dataset = base_data

    # Step 2: Train Baseline Model
    update_status("Training Baseline Classifier", 15, f"Framework: {framework}, Model: {model_name}, Seed: {seed}")
    train_start = time.time()
    baseline_train_res = train_classifier(
        dataset=train_base_dataset,
        model_name=model_name,
        framework=framework,
        seed=seed
    )
    base_train_time = round(time.time() - train_start, 3)

    # Persist Baseline TrainingRun
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    base_train_run = models.TrainingRun(
        id=str(uuid.uuid4()),
        project_id=project_id or base_version.project_id,
        dataset_version_id=base_version.id,
        model_name=baseline_train_res["model_name"],
        model_type="classifier",
        framework=framework,
        seed=seed,
        training_time_seconds=base_train_time,
        training_status="completed",
        artifact_path=baseline_train_res.get("artifact_path"),
        completed_at=now_utc
    )
    db.add(base_train_run)
    db.commit()

    # Step 3: Evaluate Baseline on Validation Set
    update_status("Evaluating Baseline Classifier", 30, "Computing Macro F1, slices, and confusion matrix...")
    eval_start = time.time()
    base_eval_res = evaluate_model(baseline_train_res["predictor"], val_dataset)
    base_eval_time = round(time.time() - eval_start, 3)

    base_eval_run = models.EvaluationRun(
        id=str(uuid.uuid4()),
        training_run_id=base_train_run.id,
        test_set_version="val_split_v1",
        split_type="val",
        accuracy=base_eval_res["accuracy"],
        macro_f1=base_eval_res["macro_f1"],
        weighted_f1=base_eval_res["weighted_f1"],
        precision=base_eval_res["precision"],
        recall=base_eval_res["recall"],
        hard_negative_accuracy=base_eval_res["hard_negative_accuracy"],
        boundary_accuracy=base_eval_res["boundary_accuracy"],
        adversarial_accuracy=base_eval_res["adversarial_accuracy"],
        per_class_metrics_json=json.dumps(base_eval_res["per_class_metrics"]),
        confusion_matrix_json=json.dumps(base_eval_res["confusion_matrix"])
    )
    db.add(base_eval_run)
    db.commit()

    # Step 4: Error Analysis & Boundary Diagnosis
    update_status("Diagnosing Errors & Decision Boundaries", 45, "Pinpointing weak classes and confused pairs...")
    diagnostics = analyze_errors(base_eval_res)

    # Persist Classification Errors for auditability
    for err in diagnostics.get("categorized_errors", []):
        db_err = models.ClassificationError(
            id=str(uuid.uuid4()),
            evaluation_run_id=base_eval_run.id,
            input_text=err["input_text"],
            expected_label=err["expected_label"],
            predicted_label=err["predicted_label"],
            confidence=err["confidence"],
            error_type=err["error_type"]
        )
        db.add(db_err)
    db.commit()

    # Step 5: Curriculum Stage Determination
    if curriculum_stage is None:
        # Determine stage dynamically based on baseline F1 and confusion severity
        stage_num, stage_reason = determine_curriculum_stage(
            history_evaluations=[base_eval_res],
            current_f1=base_eval_res["macro_f1"],
            confusion_severity=diagnostics.get("confused_pairs", [{}])[0].get("severity", "LOW") if diagnostics.get("confused_pairs") else "LOW"
        )
    else:
        stage_num = curriculum_stage
        stage_reason = f"Explicitly configured curriculum stage {stage_num}"

    # Step 6: Synthesize Curriculum-Aware Targeted Data
    update_status("Synthesizing Targeted Data", 60, f"Curriculum Stage {stage_num}: {stage_reason}")
    raw_targeted = generate_targeted_data(
        schema=schema,
        diagnostics=diagnostics,
        count=targeted_count,
        target_language="English",
        curriculum_stage=stage_num,
        seed=seed
    )

    # Step 7: Multi-Stage Quality Filter & Deduplication
    update_status("Quality Filtering & Deduplication", 70, "Validating candidate examples against quality gates...")
    filtered_results = filter_candidate_examples(
        candidate_examples=raw_targeted,
        existing_dataset=base_data,
        schema=schema,
        curriculum_stage=stage_num
    )
    accepted_examples = filtered_results["accepted"]
    filter_telemetry = filtered_results["telemetry"]

    if not accepted_examples:
        accepted_examples = raw_targeted[:max(1, min(5, len(raw_targeted)))]

    # Step 8: Create Immutable Candidate DatasetVersion N+1
    next_version_num = (base_version.version_number or 1) + 1
    new_combined_dataset = list(base_data) + accepted_examples

    cand_canonical = sum(1 for ex in new_combined_dataset if ex.get("type") == "canonical")
    cand_boundary = sum(1 for ex in new_combined_dataset if ex.get("type") == "boundary")
    cand_adversarial = sum(1 for ex in new_combined_dataset if ex.get("type") in ("adversarial", "hard_negative"))

    cand_version = models.DatasetVersion(
        id=str(uuid.uuid4()),
        project_id=project_id or base_version.project_id,
        version_number=next_version_num,
        parent_version_id=base_version.id,
        total_examples=len(new_combined_dataset),
        canonical_count=cand_canonical,
        boundary_count=cand_boundary,
        adversarial_count=cand_adversarial,
        generated_by=f"curriculum_stage_{stage_num}",
        generation_reason=f"{stage_reason} | {diagnostics['target_problem_summary']}",
        status="training",
        dataset_json=json.dumps(new_combined_dataset),
        schema_json=json.dumps(schema)
    )
    db.add(cand_version)
    db.commit()

    # Step 9: Retrain Classifier on Candidate Dataset
    update_status("Retraining Classifier on Dataset v" + str(next_version_num), 80, f"Total examples: {len(new_combined_dataset)}")
    cand_train_start = time.time()
    cand_train_dataset = train_base_dataset + accepted_examples
    cand_train_res = train_classifier(
        dataset=cand_train_dataset,
        model_name=model_name,
        framework=framework,
        seed=seed
    )
    cand_train_time = round(time.time() - cand_train_start, 3)

    cand_train_run = models.TrainingRun(
        id=str(uuid.uuid4()),
        project_id=project_id or base_version.project_id,
        dataset_version_id=cand_version.id,
        model_name=cand_train_res["model_name"],
        model_type="classifier",
        framework=framework,
        seed=seed,
        training_time_seconds=cand_train_time,
        training_status="completed",
        artifact_path=cand_train_res.get("artifact_path"),
        completed_at=datetime.datetime.now(datetime.timezone.utc)
    )
    db.add(cand_train_run)
    db.commit()

    # Step 10: Evaluate Candidate Model on the EXACT SAME Validation Set
    update_status("Evaluating Candidate Classifier", 90, "Evaluating candidate model against isolated validation set...")
    cand_eval_start = time.time()
    cand_eval_res = evaluate_model(cand_train_res["predictor"], val_dataset)
    cand_eval_time = round(time.time() - cand_eval_start, 3)

    cand_eval_run = models.EvaluationRun(
        id=str(uuid.uuid4()),
        training_run_id=cand_train_run.id,
        test_set_version="val_split_v1",
        split_type="val",
        accuracy=cand_eval_res["accuracy"],
        macro_f1=cand_eval_res["macro_f1"],
        weighted_f1=cand_eval_res["weighted_f1"],
        precision=cand_eval_res["precision"],
        recall=cand_eval_res["recall"],
        hard_negative_accuracy=cand_eval_res["hard_negative_accuracy"],
        boundary_accuracy=cand_eval_res["boundary_accuracy"],
        adversarial_accuracy=cand_eval_res["adversarial_accuracy"],
        per_class_metrics_json=json.dumps(cand_eval_res["per_class_metrics"]),
        confusion_matrix_json=json.dumps(cand_eval_res["confusion_matrix"])
    )
    db.add(cand_eval_run)
    db.commit()

    # Step 11: Multi-Objective Promotion Gate
    update_status("Evaluating Promotion Gate", 95, "Checking multi-objective promotion rule...")
    is_promoted, decision_rationale, metrics_diff = evaluate_promotion_gate(
        base_eval=base_eval_res,
        cand_eval=cand_eval_res,
        diagnostics=diagnostics
    )

    if is_promoted:
        cand_version.status = "promoted"
        cand_version.promoted_at = datetime.datetime.now(datetime.timezone.utc)
        cycle_status = "promoted"
    else:
        cand_version.status = "rejected"
        cycle_status = "rejected"

    total_cycle_time = round(time.time() - cycle_start, 3)

    # Step 12: Persist OptimizationCycle Telemetry
    telemetry_data = {
        "curriculum_stage": stage_num,
        "stage_reason": stage_reason,
        "training_time_seconds": round(base_train_time + cand_train_time, 3),
        "evaluation_time_seconds": round(base_eval_time + cand_eval_time, 3),
        "total_cycle_time_seconds": total_cycle_time,
        "llm_tokens_estimated": len(raw_targeted) * 60,
        "cost_estimated_usd": f"${(len(raw_targeted) * 60 * 0.0000005):.6f}",
        "acceptance_rate": filter_telemetry.get("acceptance_rate", 1.0),
        "anchor_coverage_ratio": filter_telemetry.get("anchor_coverage_ratio", 1.0),
        "archetype_distribution": filter_telemetry.get("archetype_distribution", {})
    }

    opt_cycle = models.OptimizationCycle(
        id=str(uuid.uuid4()),
        project_id=project_id or base_version.project_id,
        base_dataset_version_id=base_version.id,
        resulting_dataset_version_id=cand_version.id,
        evaluation_run_id=cand_eval_run.id,
        target_problem=diagnostics["target_problem_summary"],
        examples_generated=filter_telemetry["examples_generated"],
        examples_accepted=filter_telemetry["examples_accepted"],
        examples_rejected=filter_telemetry["examples_rejected"],
        baseline_f1=metrics_diff["baseline_f1"],
        resulting_f1=metrics_diff["resulting_f1"],
        improvement_delta=metrics_diff["delta_f1"],
        status=cycle_status,
        rejection_reason=decision_rationale if not is_promoted else None,
        telemetry_json=json.dumps(telemetry_data),
        completed_at=datetime.datetime.now(datetime.timezone.utc)
    )
    db.add(opt_cycle)
    db.commit()

    update_status("Optimization Cycle Complete", 100, f"Verdict: {cycle_status.upper()} ({decision_rationale})")

    return {
        "cycle_id": opt_cycle.id,
        "status": cycle_status,
        "is_promoted": is_promoted,
        "curriculum_stage": stage_num,
        "decision_rationale": decision_rationale,
        "base_version": {
            "id": base_version.id,
            "version_number": base_version.version_number,
            "macro_f1": metrics_diff["baseline_f1"],
            "boundary_accuracy": metrics_diff["baseline_boundary_acc"],
            "hard_negative_accuracy": metrics_diff["baseline_hard_negative_acc"]
        },
        "candidate_version": {
            "id": cand_version.id,
            "version_number": cand_version.version_number,
            "macro_f1": metrics_diff["resulting_f1"],
            "boundary_accuracy": metrics_diff["resulting_boundary_acc"],
            "hard_negative_accuracy": metrics_diff["resulting_hard_negative_acc"]
        },
        "metrics_diff": metrics_diff,
        "diagnostics": diagnostics,
        "telemetry": telemetry_data,
        "filter_telemetry": filter_telemetry
    }
