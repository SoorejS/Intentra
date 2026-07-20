import os
import json
import time
from anthropic import Anthropic
from openai import OpenAI

def generate_full_dataset(schema: dict, dataset_size: int) -> list:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    local_key = os.environ.get("LOCAL_API_KEY")
    local_base = os.environ.get("LOCAL_BASE_URL")
    local_model = os.environ.get("LOCAL_MODEL_NAME", "local-model")
    
    if not anthropic_key and not local_key:
        return [
            {
                "text": "This is a mock example because no API key was provided.",
                "label": "Mock Label",
                "type": "canonical",
                "typeLabel": "Canonical",
                "difficulty": "Low"
            }
        ]
        
    prompt = f"""
    Generate a dataset of exactly 10 examples for this intent schema:
    {json.dumps(schema)}
    
    Include a mix of 'canonical' (clear), 'boundary' (edge-case), and 'adversarial' (tricky) examples.
    Assign a difficulty of Low, Medium, High, or Very High.
    
    Respond with ONLY a valid JSON array matching this exact structure, do not include any other text:
    [
        {{
            "text": "The example text",
            "label": "The output class",
            "type": "canonical", 
            "typeLabel": "Canonical",
            "difficulty": "Low"
        }}
    ]
    NOTE: 'type' MUST be exactly one of: canonical, boundary, adversarial
    NOTE: 'typeLabel' MUST be exactly one of: Canonical, Boundary, Adversarial
    """
    
    dataset = []
    
    # We batch generate in chunks of 10 to prevent local models from OOM/timeout
    target_batches = max(1, dataset_size // 10)
    
    for _ in range(target_batches):
        try:
            if local_key and local_base:
                client = OpenAI(base_url=local_base, api_key=local_key, timeout=7200.0)
                response = client.chat.completions.create(
                    model=local_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=4000
                )
                content = response.choices[0].message.content
            else:
                client = Anthropic(api_key=anthropic_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            batch = json.loads(content)
            if isinstance(batch, list):
                dataset.extend(batch)
            time.sleep(1)
        except Exception as e:
            print(f"Error parsing dataset JSON batch: {e}")
            continue
            
    return dataset[:dataset_size]
