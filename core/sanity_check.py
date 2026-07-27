"""
Deduplication and sanity checking utilities for Intentra datasets.
"""

def jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute Jaccard similarity on character n-grams."""
    def ngrams(text, n):
        text = text.lower().strip()
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    a = ngrams(text_a, n)
    b = ngrams(text_b, n)
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def deduplicate_dataset(dataset: list, threshold: float = 0.85) -> tuple[list, int]:
    """
    Remove near-duplicate examples using Jaccard similarity on character trigrams.
    Returns (deduplicated_list, num_removed).
    """
    unique = []
    removed = 0
    for candidate in dataset:
        text = candidate.get("text", "")
        is_dup = any(
            jaccard_similarity(text, existing.get("text", "")) >= threshold
            for existing in unique
        )
        if is_dup:
            removed += 1
        else:
            unique.append(candidate)
    return unique, removed


def validate_labels(dataset: list, schema: dict) -> tuple[list, int]:
    """
    Remove examples whose label is not in the schema's output_classes.
    Returns (clean_list, num_removed).
    """
    valid_labels = {
        c.get("label", "").strip().lower()
        for c in schema.get("output_classes", [])
    }
    if not valid_labels:
        return dataset, 0

    clean = []
    removed = 0
    for ex in dataset:
        label = ex.get("label", "").strip().lower()
        if label in valid_labels:
            clean.append(ex)
        else:
            removed += 1
    return clean, removed


def run_sanity_check(dataset: list, schema: dict) -> dict:
    """
    Full sanity pipeline: validate labels then deduplicate.
    Returns cleaned dataset + a summary report dict.
    """
    after_validation, invalid_count = validate_labels(dataset, schema)
    after_dedup, dup_count = deduplicate_dataset(after_validation)

    return {
        "clean_dataset": after_dedup,
        "report": {
            "original_count": len(dataset),
            "invalid_labels_removed": invalid_count,
            "duplicates_removed": dup_count,
            "final_count": len(after_dedup)
        }
    }
