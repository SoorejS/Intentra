import os
import json
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()

# Objective: Identify logical fallacies disguised as reasoning.
# Classes: "Valid Reasoning", "Logical Fallacy"

INTENTRA_PROMPT = """
You are an expert AI dataset generator. Generate a dataset of exactly 10 examples for the objective: "Identify logical fallacies disguised as reasoning".
The valid output labels are exactly two: "Valid Reasoning" and "Logical Fallacy".

Include a mix of:
- 'canonical' (clear, standard examples)
- 'boundary' (edge-cases that are hard to distinguish)
- 'adversarial' (tricky examples designed to fool a naive model, e.g., formal language but flawed logic, or emotional language but valid logic).

Respond with ONLY a valid JSON array matching this exact structure, do not include any other text:
[
    {
        "text": "The example text",
        "label": "Valid Reasoning or Logical Fallacy",
        "type": "canonical | boundary | adversarial"
    }
]
"""

NAIVE_PROMPT = """
Generate examples of logical fallacies and valid reasoning.
The valid output labels are exactly two: "Valid Reasoning" and "Logical Fallacy".

Respond with ONLY a valid JSON array matching this exact structure, do not include any other text:
[
    {
        "text": "The example text",
        "label": "Valid Reasoning or Logical Fallacy"
    }
]
Generate exactly 10 examples.
"""

def get_word_count(dataset):
    return sum(len(ex["text"].split()) for ex in dataset)

def generate_dataset(prompt):
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    local_key = os.environ.get("LOCAL_API_KEY")
    local_base = os.environ.get("LOCAL_BASE_URL")
    local_model = os.environ.get("LOCAL_MODEL_NAME", "local-model")
    
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
        elif anthropic_key:
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        else:
            print("No API key found in environment (.env).")
            return []

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        try:
            return json.loads(content)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw content was:\n{content}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LOCAL_API_KEY"):
        print("Neither ANTHROPIC_API_KEY nor LOCAL_API_KEY found.")
        return

    print("Generating Intentra dataset in batches of 10...")
    intentra_data = []
    
    while len(intentra_data) < 100:
        batch = generate_dataset(INTENTRA_PROMPT)
        if not batch:
            print("Failed to generate intentra dataset batch. Skipping or retrying.")
            continue
        intentra_data.extend(batch)
        print(f"Intentra progress: {len(intentra_data)}/100 examples.")
        time.sleep(1)

    intentra_words = get_word_count(intentra_data)
    print(f"Intentra Dataset Completed: {len(intentra_data)} examples, {intentra_words} words.")

    print("Generating Naive dataset...")
    naive_data = []
    naive_words = 0
    # Keep generating naive examples until we match or exceed the intentra word count
    # This controls for the confound of token volume.
    while naive_words < intentra_words:
        batch = generate_dataset(NAIVE_PROMPT)
        if not batch:
            break
        for ex in batch:
            naive_data.append(ex)
            naive_words += len(ex["text"].split())
            if naive_words >= intentra_words:
                break
        print(f"Naive Dataset accumulation: {len(naive_data)} examples, {naive_words} words.")
        time.sleep(1) # rate limit prevention

    # Save datasets
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "intentra_train.json"), "w") as f:
        json.dump(intentra_data, f, indent=4)
        
    with open(os.path.join(os.path.dirname(__file__), "naive_train.json"), "w") as f:
        json.dump(naive_data, f, indent=4)

    print("Datasets generated and saved successfully.")

if __name__ == "__main__":
    main()
