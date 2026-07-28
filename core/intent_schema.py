import json
from core.llm_client import call_llm, extract_json

def generate_intent_schema(objective: str, domain_hint: str) -> dict:
    prompt = f"""
You are an expert NLP data engineer. Create an intent classification schema for the objective: '{objective}'.
Domain hint: {domain_hint}

Respond with ONLY a valid JSON object matching this structure:
{{
    "task_type": "Multi-class Intent Classification",
    "output_classes": [
        {{"label": "Class1", "description": "..."}},
        {{"label": "Class2", "description": "..."}}
    ],
    "pragmatic_signals": ["Signal1", "Signal2", "Signal3"],
    "why_existing_tools_fail": "Brief explanation of why keyword matching fails here."
}}
"""

    content = call_llm(prompt, max_tokens=1000, temperature=0.2)
    schema = extract_json(content)

    # Validate the schema has required fields
    if not isinstance(schema, dict):
        raise ValueError(f"Expected dict from LLM, got {type(schema).__name__}")
    if "output_classes" not in schema or not schema["output_classes"]:
        raise ValueError("Schema missing 'output_classes'")

    return schema
