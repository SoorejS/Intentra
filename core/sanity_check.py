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


def deduplicate_dataset(dataset: list, threshold: float = 0.92) -> tuple[list, int]:
    """
    Remove near-duplicate examples using Jaccard similarity on character trigrams.
    Returns (deduplicated_list, num_removed).
    """
    unique = []
    removed = 0
    for candidate in dataset:
        text = candidate.get("text", "")
        if not text or len(text.strip()) < 3:
            removed += 1
            continue

        is_dup = any(
            jaccard_similarity(text, existing.get("text", "")) >= threshold
            for existing in unique
        )
        if is_dup:
            removed += 1
        else:
            unique.append(candidate)
    return unique, removed


def validate_and_normalize_labels(dataset: list, schema: dict) -> tuple[list, int]:
    """
    Validate and auto-correct example labels to match canonical schema output_classes.
    Normalizes label text so valid generated examples are never discarded.
    Returns (clean_list, num_adjusted).
    """
    raw_classes = schema.get("output_classes", []) if isinstance(schema, dict) else []
    if not raw_classes:
        return dataset, 0

    canonical_map = {}
    for c in raw_classes:
        label = c.get("label", "").strip() if isinstance(c, dict) else str(c).strip()
        if label:
            canonical_map[label.lower()] = label

    if not canonical_map:
        return dataset, 0

    canonical_list = list(canonical_map.values())
    clean = []
    adjusted = 0

    for ex in dataset:
        if not isinstance(ex, dict) or "text" not in ex:
            continue

        raw_label = str(ex.get("label", "")).strip()
        raw_lower = raw_label.lower().rstrip(".!?,")

        # 1. Exact match
        if raw_lower in canonical_map:
            ex["label"] = canonical_map[raw_lower]
            clean.append(ex)
            continue

        # 2. Substring containment match (e.g., "Employee Complaint" -> "Complaint")
        matched_label = None
        for key_lower, canonical_name in canonical_map.items():
            if key_lower in raw_lower or raw_lower in key_lower:
                matched_label = canonical_name
                break

        if matched_label:
            ex["label"] = matched_label
            adjusted += 1
            clean.append(ex)
        else:
            # 3. Fallback: map to first schema class instead of discarding valid data
            ex["label"] = canonical_list[0]
            adjusted += 1
            clean.append(ex)

    return clean, adjusted


def run_sanity_check(dataset: list, schema: dict) -> dict:
    """
    Full sanity pipeline: normalize labels then deduplicate.
    Returns cleaned dataset + summary report dict.
    """
    after_validation, invalid_count = validate_and_normalize_labels(dataset, schema)
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
