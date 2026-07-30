import json
from core.llm_client import call_llm, extract_json

def generate_intent_schema(objective: str, domain_hint: str, *,
                            target_language: str = "English",
                            is_multilabel: bool = False,
                            custom_classes: list = None) -> dict:
    classification_type = "Multi-label Intent Classification" if is_multilabel else "Multi-class Intent Classification"

    custom_classes_instruction = ""
    if custom_classes and len(custom_classes) >= 2:
        classes_str = ", ".join(f'"{c}"' for c in custom_classes)
        custom_classes_instruction = f"""
IMPORTANT: Use EXACTLY these output class labels: [{classes_str}].
Do NOT invent new class names. Use the provided labels verbatim."""

    multilabel_note = ""
    if is_multilabel:
        multilabel_note = """
This is a MULTI-LABEL task. Each example can belong to one or more classes simultaneously.
In the output_classes, describe classes that can co-occur."""

    prompt = f"""
You are an expert NLP data engineer. Create an intent classification schema for the objective: '{objective}'.
Domain hint: {domain_hint}
Target language for all text content: {target_language}
Task type: {classification_type}
{custom_classes_instruction}
{multilabel_note}

Respond with ONLY a valid JSON object matching this structure:
{{
    "task_type": "{classification_type}",
    "output_classes": [
        {{"label": "Class1", "description": "..."}},
        {{"label": "Class2", "description": "..."}}
    ],
    "pragmatic_signals": ["Signal1", "Signal2", "Signal3"],
    "why_existing_tools_fail": "Brief explanation of why keyword matching fails here.",
    "target_language": "{target_language}"
}}
"""

    content = call_llm(prompt, max_tokens=1000, temperature=0.2)
    schema = extract_json(content)

    # Validate the schema has required fields
    if not isinstance(schema, dict):
        raise ValueError(f"Expected dict from LLM, got {type(schema).__name__}")
    if "output_classes" not in schema or not schema["output_classes"]:
        raise ValueError("Schema missing 'output_classes'")

    # Ensure target_language is preserved
    schema["target_language"] = target_language
    schema["is_multilabel"] = is_multilabel

    return schema
