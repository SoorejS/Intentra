import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def prepare_dataset(data, tokenizer):
    # Mapping labels to integers
    label_map = {"Valid Reasoning": 0, "Logical Fallacy": 1}
    
    texts = [item["text"] for item in data]
    labels = [label_map[item["label"]] for item in data]
    
    hf_dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_and_eval(train_dataset, test_dataset, seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_data = prepare_dataset(train_dataset, tokenizer)
    test_data = prepare_dataset(test_dataset, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"./results/seed_{seed}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        seed=seed,
        logging_steps=10,
        report_to="none" # disable wandb logging
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result

def main():
    base_dir = os.path.dirname(__file__)
    intentra_path = os.path.join(base_dir, "intentra_train.json")
    naive_path = os.path.join(base_dir, "naive_train.json")
    test_path = os.path.join(base_dir, "hand_written_test_set.json")
    
    if not os.path.exists(intentra_path) or not os.path.exists(naive_path):
        print("Training datasets not found! Run benchmark_generation.py first.")
        return

    intentra_data = load_data(intentra_path)
    naive_data = load_data(naive_path)
    test_data = load_data(test_path)
    
    seeds = [42, 123, 999]
    
    print("="*50)
    print("Evaluating NAIVE Dataset")
    print("="*50)
    naive_results = []
    for s in seeds:
        print(f"--- Training with seed {s} ---")
        res = train_and_eval(naive_data, test_data, s)
        naive_results.append(res['eval_f1'])
        print(f"Seed {s} F1: {res['eval_f1']:.4f}")
        
    print("="*50)
    print("Evaluating INTENTRA Dataset")
    print("="*50)
    intentra_results = []
    for s in seeds:
        print(f"--- Training with seed {s} ---")
        res = train_and_eval(intentra_data, test_data, s)
        intentra_results.append(res['eval_f1'])
        print(f"Seed {s} F1: {res['eval_f1']:.4f}")
        
    print("\n" + "="*50)
    print("FINAL RESULTS (F1 Score on Hard Test Set)")
    print("="*50)
    print(f"NAIVE    | Mean F1: {np.mean(naive_results):.4f} | Std: {np.std(naive_results):.4f} | Runs: {naive_results}")
    print(f"INTENTRA | Mean F1: {np.mean(intentra_results):.4f} | Std: {np.std(intentra_results):.4f} | Runs: {intentra_results}")
    print("="*50)

if __name__ == "__main__":
    main()
