import json
import time
from core.llm_client import call_llm, extract_json

def generate_full_dataset(schema: dict, dataset_size: int, on_batch_callback=None, *,
                           target_language: str = "English",
                           is_multilabel: bool = False,
                           refinement_instruction: str = None) -> list:

    language_instruction = ""
    if target_language and target_language != "English":
        language_instruction = f"\nIMPORTANT: Generate ALL example text content in {target_language}. Labels should remain in English."

    multilabel_instruction = ""
    if is_multilabel or schema.get("is_multilabel"):
        multilabel_instruction = """
This is a MULTI-LABEL task. Some examples should have multiple applicable labels.
For multi-label examples, use a comma-separated string for the 'label' field, e.g. "Complaint, Urgent".
"""

    refinement_note = ""
    if refinement_instruction:
        refinement_note = f"""
REFINEMENT INSTRUCTION: {refinement_instruction}
Focus on generating examples that match this specific refinement request.
"""

    prompt = f"""
Generate a dataset of exactly 10 training examples for this intent classification schema:
{json.dumps(schema)}
{language_instruction}
{multilabel_instruction}
{refinement_note}
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
                if callable(on_batch_callback):
                    try:
                        on_batch_callback(batch_num + 1, target_batches, batch)
                    except Exception as cb_err:
                        print(f"[dataset_generator] Callback error: {cb_err}")

            # Small pause between batches to respect rate limits
            if batch_num < target_batches - 1:
                time.sleep(0.5)
        except Exception as e:
            print(f"[dataset_generator] Batch {batch_num + 1} failed: {e}")
            continue

    return dataset[:dataset_size]
