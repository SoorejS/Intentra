import os
import json
from anthropic import Anthropic

def generate_full_dataset(schema: dict, dataset_size: int) -> list:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [
            {
                "text": "This is a mock example because no API key was provided.",
                "label": "Mock Label",
                "type": "canonical",
                "typeLabel": "Canonical",
                "difficulty": "Low"
            }
        ]
        
    client = Anthropic(api_key=api_key)
    
    prompt = f"""
    Generate a dataset of exactly {dataset_size} examples for this intent schema:
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
    
    try:
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
            
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing dataset JSON: {e}")
        return []
