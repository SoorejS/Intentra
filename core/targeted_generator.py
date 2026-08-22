"""
Intentra V2.1 - Curriculum-Aware Targeted Data Generator
Synthesizes canonical anchors, controlled variations, boundary examples, and hard negatives
conditioned on curriculum stage, diagnosed classifier confusion pairs, and weak classes.
"""

import json
from core.llm_client import call_llm, extract_json


import numpy as np


def generate_targeted_data(
    schema: dict,
    diagnostics: dict,
    count: int = 30,
    target_language: str = "English",
    curriculum_stage: int = 3,
    archetype_mix: dict | None = None,
    seed: int = 42
) -> list:
    """
    Generate targeted synthetic examples with full curriculum awareness and provenance.
    """
    primary_pair = diagnostics.get("primary_focus_pair")
    weak_class = diagnostics.get("primary_focus_weak_class")
    target_problem = diagnostics.get("target_problem_summary", "Decision boundary improvement")

    classes = [c.get("label") for c in schema.get("output_classes", [])] if isinstance(schema, dict) else []
    if not classes:
        classes = [primary_pair[0], primary_pair[1]] if primary_pair else ["Class A", "Class B"]

    cls_a = primary_pair[0] if primary_pair else (classes[0] if len(classes) > 0 else "Class A")
    cls_b = primary_pair[1] if primary_pair else (classes[1] if len(classes) > 1 else "Class B")

    pair_list = [cls_a, cls_b] if primary_pair else None
    reason_str = f"Curriculum Stage {curriculum_stage} targeting '{cls_a}' vs '{cls_b}'"

    # Stage-aware instruction
    if curriculum_stage == 1:
        stage_desc = "Stage 1 Anchor Grounding: Generate 100% crystal-clear, unambiguous canonical examples."
    elif curriculum_stage == 2:
        stage_desc = "Stage 2 Controlled Variation: Generate diverse paraphrases and stylistic variants with clear labels."
    elif curriculum_stage == 3:
        stage_desc = f"Stage 3 Boundary Disambiguation: Generate boundary examples directly on the decision boundary between '{cls_a}' and '{cls_b}'."
    else:
        stage_desc = f"Stage 4 Contrastive Hardening: Generate subtle contrastive and hard-negative cases separating '{cls_a}' from '{cls_b}'."

    prompt = f"""You are Intentra V2.1's Curriculum-Aware Targeted Generator.
Current Optimization Stage: {stage_desc}
Diagnostic Context: "{target_problem}"

Target Classes:
- Class A: "{cls_a}"
- Class B: "{cls_b}"

Generate EXACTLY {count} synthetic training examples in {target_language}.
VALID OUTPUT CLASSES: {json.dumps(classes)}

OUTPUT FORMAT (JSON ARRAY ONLY, NO PREAMBLE):
[
  {{
    "text": "Clear realistic statement",
    "label": "{cls_a}",
    "type": "boundary",
    "archetype": "boundary_case",
    "difficulty": "MEDIUM_HARD",
    "difficulty_rationale": "Clarifies boundary between {cls_a} and {cls_b}"
  }}
]
"""

    try:
        response_text = call_llm(prompt, max_tokens=3000, temperature=0.75)
        raw_examples = extract_json(response_text)
        if isinstance(raw_examples, dict) and "examples" in raw_examples:
            raw_examples = raw_examples["examples"]
        elif not isinstance(raw_examples, list):
            raw_examples = []

        validated_examples = []
        for item in raw_examples:
            if isinstance(item, dict) and "text" in item and "label" in item:
                lbl = item["label"].strip()
                matched_lbl = next((c for c in classes if c.lower() == lbl.lower()), lbl)
                itype = item.get("type", "boundary" if curriculum_stage >= 3 else "canonical")
                arch = item.get("archetype", "boundary_case" if itype == "boundary" else "hard_negative" if itype == "hard_negative" else "canonical_anchor")
                
                validated_examples.append({
                    "text": item["text"].strip(),
                    "label": matched_lbl,
                    "type": itype,
                    "generation_stage": curriculum_stage,
                    "archetype": arch,
                    "target_class": matched_lbl,
                    "confusion_pair": pair_list,
                    "difficulty_rationale": item.get("difficulty_rationale", reason_str),
                    "generation_reason": reason_str,
                    "difficulty": item.get("difficulty", "MEDIUM_HARD" if curriculum_stage == 3 else "HARD" if curriculum_stage == 4 else "EASY"),
                    "source_error_ids": [],
                    "generated_by": "targeted_curriculum_v21"
                })

        if validated_examples:
            return validated_examples

    except Exception as e:
        print(f"[Targeted Generator] LLM notice ({e}), using curriculum fallback synthesis.")

    # Programmatic Fallback for offline execution / test determinism
    return _generate_programmatic_curriculum_fallback(cls_a, cls_b, count, reason_str, curriculum_stage, pair_list, seed=seed)


def _generate_programmatic_curriculum_fallback(
    cls_a: str,
    cls_b: str,
    count: int,
    reason: str,
    curriculum_stage: int,
    pair_list: list | None,
    seed: int = 42
) -> list:
    """Generate deterministic curriculum-structured examples offline with seed-dependent sampling."""
    rng = np.random.default_rng(seed)
    if curriculum_stage == 1:
        templates = [
            (f"This is explicitly and strictly a request regarding {cls_a}.", cls_a, "canonical", "canonical_anchor", "EASY"),
            (f"I need direct assistance with {cls_a} for my account.", cls_a, "canonical", "canonical_anchor", "EASY"),
            (f"Please help me complete my {cls_b} immediately.", cls_b, "canonical", "canonical_anchor", "EASY"),
            (f"I am contacting support exclusively for {cls_b}.", cls_b, "canonical", "canonical_anchor", "EASY")
        ]
    elif curriculum_stage == 2:
        templates = [
            (f"Hello, I would like to inquire about the standard procedure for {cls_a}.", cls_a, "canonical", "controlled_variation", "EASY_MEDIUM"),
            (f"Can someone from your support team guide me through {cls_a}?", cls_a, "canonical", "controlled_variation", "EASY_MEDIUM"),
            (f"Good morning, please assist with finalizing my {cls_b}.", cls_b, "canonical", "controlled_variation", "EASY_MEDIUM"),
            (f"Regarding my recent status: kindly process this as {cls_b}.", cls_b, "canonical", "controlled_variation", "EASY_MEDIUM")
        ]
    elif curriculum_stage == 3:
        templates = [
            (f"I need to know the process for {cls_a}, but please clarify how it impacts my recent {cls_b}.", cls_a, "boundary", "boundary_case", "MEDIUM_HARD"),
            (f"Can I proceed with {cls_a} even if the system already shows a status for {cls_b}?", cls_a, "boundary", "boundary_case", "MEDIUM_HARD"),
            (f"Regarding my {cls_b} request earlier: I actually want to formally change it to a {cls_a}.", cls_a, "boundary", "boundary_case", "MEDIUM_HARD"),
            (f"I was originally looking into {cls_a}, but right now my immediate priority is {cls_b}.", cls_b, "boundary", "boundary_case", "MEDIUM_HARD"),
            (f"If {cls_a} is not possible at this time, please proceed directly with {cls_b}.", cls_b, "boundary", "boundary_case", "MEDIUM_HARD")
        ]
    else:
        templates = [
            (f"Do not process this as a {cls_b}; I am explicitly demanding a {cls_a} instead.", cls_a, "hard_negative", "hard_negative", "HARD"),
            (f"The portal showed options for {cls_b}, but my true requirement is {cls_a}.", cls_a, "hard_negative", "hard_negative", "HARD"),
            (f"Although I previously discussed {cls_a}, finalize my {cls_b} right now.", cls_b, "hard_negative", "hard_negative", "HARD"),
            (f"Forget any prior mention of {cls_a}; this is strictly a {cls_b}.", cls_b, "hard_negative", "hard_negative", "HARD")
        ]

    results = []
    for _ in range(count):
        idx = int(rng.integers(0, len(templates)))
        tmpl_text, tmpl_lbl, tmpl_type, tmpl_arch, tmpl_diff = templates[idx]
        ref_id = int(rng.integers(1000, 9999))
        variant_text = f"{tmpl_text} (Case #{ref_id})"
        results.append({
            "text": variant_text,
            "label": tmpl_lbl,
            "type": tmpl_type,
            "generation_stage": curriculum_stage,
            "archetype": tmpl_arch,
            "target_class": tmpl_lbl,
            "confusion_pair": pair_list,
            "difficulty_rationale": f"Curriculum Stage {curriculum_stage} {tmpl_arch} between {cls_a} and {cls_b}",
            "generation_reason": reason,
            "difficulty": tmpl_diff,
            "source_error_ids": [],
            "generated_by": "targeted_curriculum_v21"
        })

    return results
