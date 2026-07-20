import os
import json
from anthropic import Anthropic
from openai import OpenAI

def generate_intent_schema(objective: str, domain_hint: str) -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    local_key = os.environ.get("LOCAL_API_KEY")
    local_base = os.environ.get("LOCAL_BASE_URL")
    local_model = os.environ.get("LOCAL_MODEL_NAME", "local-model")
    
    if not anthropic_key and not local_key:
        # Fallback to a mock response if no key is found so the demo doesn't crash completely
        return {
            "task_type": "Multi-class Intent Classification",
            "output_classes": [{"label": "Urgent", "description": "Needs immediate attention"}, {"label": "Normal", "description": "Standard processing"}],
            "pragmatic_signals": ["Urgency", "Frustration"]
        }
        
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
        "pragmatic_signals": ["Signal1", "Signal2", "Signal3"]
    }}
    """
    
    try:
        if local_key and local_base:
            client = OpenAI(base_url=local_base, api_key=local_key, timeout=7200.0)
            response = client.chat.completions.create(
                model=local_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            content = response.choices[0].message.content
        else:
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {
            "task_type": "Error",
            "output_classes": [{"label": "Error"}],
            "pragmatic_signals": ["Error"]
        }
