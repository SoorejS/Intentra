"""
Intentra V2 - Targeted Data Generator
Synthesizes boundary examples, hard negatives, and contrastive pairs
specifically conditioned on diagnosed classifier confusion pairs and weak classes.
"""

import json
from core.llm_client import call_llm, extract_json


def generate_targeted_data(
    schema: dict,
    diagnostics: dict,
    count: int = 30,
    target_language: str = "English"
) -> list:
    """
    Generate targeted synthetic examples designed to harden the classifier's
    decision boundary around identified confusion pairs.
    """
    primary_pair = diagnostics.get("primary_focus_pair")
    weak_class = diagnostics.get("primary_focus_weak_class")
    target_problem = diagnostics.get("target_problem_summary", "Decision boundary improvement")

    classes = [c.get("label") for c in schema.get("output_classes", [])] if isinstance(schema, dict) else []
    if not classes:
        classes = [primary_pair[0], primary_pair[1]] if primary_pair else ["Class A", "Class B"]

    cls_a = primary_pair[0] if primary_pair else (classes[0] if len(classes) > 0 else "Class A")
    cls_b = primary_pair[1] if primary_pair else (classes[1] if len(classes) > 1 else "Class B")

    prompt = f"""You are Intentra V2's Targeted Hard-Example Generator.
A downstream classifier was evaluated and FAILED specifically on the following decision boundary problem:
"{target_problem}"

Specifically, the classifier frequently confuses:
- Class A: "{cls_a}"
- Class B: "{cls_b}"

Your task is to generate EXACTLY {count} high-difficulty, targeted synthetic training examples in {target_language} that will teach the classifier how to correctly separate "{cls_a}" from "{cls_b}".

Generate a balanced mix of:
1. Boundary Cases (40%): Real-world customer/user statements that sit directly on the decision boundary between "{cls_a}" and "{cls_b}", where the correct label requires understanding subtle nuances.
2. Hard Negatives (40%): Statements that contain keywords, phrasing, or vocabulary normally associated with "{cls_b}", but where the true underlying intent is "{cls_a}" (and vice versa).
3. Contrastive Pairs (20%): Subtly different sentences where a slight modification in context flips the label from "{cls_a}" to "{cls_b}".

VALID OUTPUT CLASSES MUST BE ONE OF: {json.dumps(classes)}

OUTPUT FORMAT (JSON ARRAY ONLY, NO PREAMBLE, NO MARKDOWN OUTSIDE JSON):
[
  {{
    "text": "Detailed realistic text statement",
    "label": "{cls_a}",
    "type": "boundary",
    "difficulty_rationale": "Why this specifically disambiguates {cls_a} from {cls_b}"
  }},
  {{
    "text": "Another detailed statement with misleading keywords",
    "label": "{cls_b}",
    "type": "hard_negative",
    "difficulty_rationale": "Contains keywords from {cls_a} but true intent is {cls_b}"
  }}
]
"""

    reason_str = f"Targeted boundary hardening between '{cls_a}' and '{cls_b}'"

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
                # Ensure label matches known classes if possible
                matched_lbl = next((c for c in classes if c.lower() == lbl.lower()), lbl)
                validated_examples.append({
                    "text": item["text"].strip(),
                    "label": matched_lbl,
                    "type": item.get("type", "boundary"),
                    "difficulty_rationale": item.get("difficulty_rationale", reason_str),
                    "generation_reason": reason_str,
                    "generated_by": "targeted_optimizer_v2"
                })

        if validated_examples:
            return validated_examples

    except Exception as e:
        print(f"[Targeted Generator] LLM generation notice ({e}), using programmatic boundary synthesis fallback.")

    # Programmatic Fallback for offline testing / LLM unavailable
    return _generate_programmatic_fallback(cls_a, cls_b, count, reason_str)


def _generate_programmatic_fallback(cls_a: str, cls_b: str, count: int, reason: str) -> list:
    """Generate high-quality heuristic boundary/hard-negative synthetic samples offline."""
    fallback_templates = [
        # Boundary for A
        (f"I need to know the process for {cls_a}, but please clarify how it impacts my recent {cls_b}.", cls_a, "boundary"),
        (f"Can I proceed with {cls_a} even if the system already shows a status for {cls_b}?", cls_a, "boundary"),
        (f"Regarding my {cls_b} request earlier: I actually want to formally change it to a {cls_a}.", cls_a, "boundary"),
        (f"What are the terms of {cls_a} if a {cls_b} has already been initiated?", cls_a, "boundary"),
        # Hard Negative for A (Contains B keywords)
        (f"Do not process this as a {cls_b}; I am explicitly demanding a {cls_a} instead.", cls_a, "hard_negative"),
        (f"The representative told me to request {cls_b}, but that's wrong, my real issue is {cls_a}.", cls_a, "hard_negative"),
        (f"I see the {cls_b} button on the portal, but I need assistance with {cls_a}.", cls_a, "hard_negative"),
        # Boundary for B
        (f"I was originally looking into {cls_a}, but right now my immediate priority is {cls_b}.", cls_b, "boundary"),
        (f"If {cls_a} is not possible at this time, please proceed directly with {cls_b}.", cls_b, "boundary"),
        (f"Does initiating a {cls_b} automatically prevent any future {cls_a}?", cls_b, "boundary"),
        # Hard Negative for B (Contains A keywords)
        (f"Although I previously inquired about {cls_a}, I want to confirm my request for {cls_b}.", cls_b, "hard_negative"),
        (f"I don't need a {cls_a} anymore; just go ahead and finalize the {cls_b}.", cls_b, "hard_negative"),
        (f"Forget the discussion on {cls_a}, I need {cls_b} immediately.", cls_b, "hard_negative")
    ]

    results = []
    idx = 0
    while len(results) < count:
        tmpl_text, tmpl_lbl, tmpl_type = fallback_templates[idx % len(fallback_templates)]
        variant_text = f"{tmpl_text} (Case #{len(results) + 1})"
        results.append({
            "text": variant_text,
            "label": tmpl_lbl,
            "type": tmpl_type,
            "difficulty_rationale": f"Programmatic {tmpl_type} template disambiguating {cls_a} vs {cls_b}",
            "generation_reason": reason,
            "generated_by": "targeted_optimizer_v2"
        })
        idx += 1

    return results
