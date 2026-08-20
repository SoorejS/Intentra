"""
Intentra V2 - Classifier Trainer Module
Supports dual-engine training:
  1. Fast Mode (sklearn TF-IDF + Logistic Regression): Sub-second training for rapid optimization loops & testing.
  2. Heavy Mode (Hugging Face DistilBERT Sequence Classification): Deep learning transformer fine-tuning.
"""

import os
import time
import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SklearnPredictor:
    """Wrapper for fast scikit-learn pipeline to provide uniform predict API."""
    def __init__(self, pipeline, label2id, id2label):
        self.pipeline = pipeline
        self.label2id = label2id
        self.id2label = id2label

    def predict(self, texts):
        if not texts:
            return [], []
        preds_raw = self.pipeline.predict(texts)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(texts)
            confidences = np.max(probs, axis=1).tolist()
        else:
            confidences = [1.0] * len(texts)
        
        preds = [p if isinstance(p, str) else self.id2label[p] for p in preds_raw]
        return preds, confidences


class TransformerPredictor:
    """Wrapper for Hugging Face sequence classification model."""
    def __init__(self, model, tokenizer, label2id, id2label, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, batch_size=16):
        if not texts:
            return [], []
        all_preds = []
        all_confs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_ids = np.argmax(probs, axis=-1)
                confs = np.max(probs, axis=-1)

            for pid, conf in zip(pred_ids, confs):
                all_preds.append(self.id2label[int(pid)])
                all_confs.append(float(conf))

        return all_preds, all_confs


def train_classifier(
    dataset: list,
    model_name: str = "distilbert-base-uncased",
    framework: str = "sklearn_fast",
    seed: int = 42,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    save_dir: str | None = None
) -> dict:
    """
    Train a text classification model on the given dataset.
    dataset: list of dicts with at least {"text": str, "label": str}
    """
    if not dataset:
        raise ValueError("Cannot train on an empty dataset")

    # Extract distinct labels
    labels = sorted(list(set(item["label"] for item in dataset if item.get("label"))))
    if len(labels) < 2:
        raise ValueError(f"Training requires at least 2 distinct classes. Found: {labels}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}

    texts = [item["text"] for item in dataset]
    targets = [item["label"] for item in dataset]

    start_time = time.time()
    artifact_path = None

    if framework == "sklearn_fast":
        # Deterministic fast TF-IDF + Logistic Regression
        np.random.seed(seed)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
            ("clf", LogisticRegression(random_state=seed, max_iter=200, C=1.0))
        ])
        pipeline.fit(texts, targets)
        predictor = SklearnPredictor(pipeline, label2id, id2label)
        training_time = time.time() - start_time

        if save_dir:
            import joblib
            os.makedirs(save_dir, exist_ok=True)
            artifact_path = os.path.join(save_dir, f"sklearn_model_seed_{seed}.joblib")
            joblib.dump(pipeline, artifact_path)

        return {
            "predictor": predictor,
            "label2id": label2id,
            "id2label": id2label,
            "framework": "sklearn_fast",
            "model_name": "tfidf_logistic_regression",
            "seed": seed,
            "training_time_seconds": round(training_time, 3),
            "artifact_path": artifact_path,
            "classes": labels
        }

    elif framework == "transformers":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset as HFDataset

        torch.manual_seed(seed)
        np.random.seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )

        numeric_labels = [label2id[lbl] for lbl in targets]
        hf_dataset = HFDataset.from_dict({"text": texts, "label": numeric_labels})

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_train = hf_dataset.map(tokenize_function, batched=True)

        out_dir = save_dir or f"./results/run_seed_{seed}_{int(time.time())}"
        os.makedirs(out_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=out_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="no",
            seed=seed,
            logging_steps=10,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
        )

        trainer.train()
        training_time = time.time() - start_time

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = TransformerPredictor(model, tokenizer, label2id, id2label, device=device)

        return {
            "predictor": predictor,
            "label2id": label2id,
            "id2label": id2label,
            "framework": "transformers",
            "model_name": model_name,
            "seed": seed,
            "training_time_seconds": round(training_time, 3),
            "artifact_path": out_dir,
            "classes": labels
        }

    else:
        raise ValueError(f"Unsupported framework: {framework}. Choose 'sklearn_fast' or 'transformers'.")
