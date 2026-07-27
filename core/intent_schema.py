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

    try:
        content = call_llm(prompt, max_tokens=1000, temperature=0.2)
        return extract_json(content)
    except Exception as e:
        print(f"[intent_schema] Error: {e}")
        return {
            "task_type": "Multi-class Intent Classification",
            "output_classes": [
                {"label": "Positive", "description": "Positive intent"},
                {"label": "Negative", "description": "Negative intent"}
            ],
            "pragmatic_signals": ["Tone", "Context", "Phrasing"],
            "why_existing_tools_fail": "Unable to generate schema - using fallback."
        }
