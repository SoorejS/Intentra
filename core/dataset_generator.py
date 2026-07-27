import json
import time
from core.llm_client import call_llm, extract_json

def generate_full_dataset(schema: dict, dataset_size: int) -> list:
    prompt = f"""
Generate a dataset of exactly 10 training examples for this intent classification schema:
{json.dumps(schema)}

Include a balanced mix of:
- 'canonical': clear, unambiguous examples
- 'boundary': edge-cases that are genuinely ambiguous
- 'adversarial': tricky examples that look like one class but are actually another

Assign a difficulty of Low, Medium, High, or Very High.

Respond with ONLY a valid JSON array. No extra text, no markdown:
[
    {{
        "text": "The example text here",
        "label": "One of the output class labels",
        "type": "canonical",
        "typeLabel": "Canonical",
        "difficulty": "Low"
    }}
]
RULES:
- 'type' MUST be exactly one of: canonical, boundary, adversarial
- 'typeLabel' MUST be exactly one of: Canonical, Boundary, Adversarial
- 'label' MUST be one of the output_classes labels from the schema
- Generate exactly 10 examples per call
"""

    dataset = []
    target_batches = max(1, dataset_size // 10)

    for batch_num in range(target_batches):
        try:
            print(f"[dataset_generator] Batch {batch_num + 1}/{target_batches}")
            content = call_llm(prompt, max_tokens=4000, temperature=0.7)
            batch = extract_json(content)
            if isinstance(batch, list):
                dataset.extend(batch)
            # Small pause between batches to respect rate limits
            if batch_num < target_batches - 1:
                time.sleep(0.5)
        except Exception as e:
            print(f"[dataset_generator] Batch {batch_num + 1} failed: {e}")
            continue

    return dataset[:dataset_size]
