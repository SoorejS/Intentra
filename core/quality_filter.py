"""
Intentra V2 - Quality Filter Module
Multi-stage quality validation gate for synthetic training examples.
Filters near-duplicates, validates schema labels, and audits rejection telemetry.
"""

def get_char_trigrams(text: str) -> set:
    """Generate set of character 3-grams for Jaccard similarity comparison."""
    text_clean = "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).strip()
    if len(text_clean) < 3:
        return {text_clean} if text_clean else set()
    return {text_clean[i:i+3] for i in range(len(text_clean) - 2)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def filter_candidate_examples(
    candidate_examples: list,
    existing_dataset: list,
    schema: dict,
    jaccard_threshold: float = 0.70
) -> dict:
    """
    Filter and validate candidate synthetic examples against quality gates.
    Returns:
      accepted: list of valid examples
      rejected: list of rejected examples with reason
      telemetry: summary counts and stats
    """
    valid_classes = [c.get("label") for c in schema.get("output_classes", [])] if isinstance(schema, dict) else []
    valid_classes_lower = {c.lower(): c for c in valid_classes} if valid_classes else {}

    existing_trigrams = [get_char_trigrams(ex.get("text", "")) for ex in existing_dataset]

    accepted = []
    rejected = []
    rejection_reasons = {}

    for cand in candidate_examples:
        text = cand.get("text", "").strip()
        label = cand.get("label", "").strip()

        # Gate 1: Length & Emptiness
        if len(text) < 10:
            reason = "Text too short (< 10 characters)"
            rejected.append({**cand, "rejection_reason": reason})
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        # Gate 2: Schema Label Validity & Normalization
        if valid_classes:
            if label.lower() in valid_classes_lower:
                label = valid_classes_lower[label.lower()]
                cand["label"] = label
            else:
                reason = f"Invalid class label '{label}' not present in schema"
                rejected.append({**cand, "rejection_reason": reason})
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

        # Gate 3: Near-Duplicate Detection (Jaccard > 0.70)
        cand_trigram = get_char_trigrams(text)
        is_duplicate = False

        # Compare with existing dataset
        for ex_trigram in existing_trigrams:
            if jaccard_similarity(cand_trigram, ex_trigram) > jaccard_threshold:
                is_duplicate = True
                break

        # Compare with already accepted batch
        if not is_duplicate:
            for acc in accepted:
                acc_trigram = get_char_trigrams(acc["text"])
                if jaccard_similarity(cand_trigram, acc_trigram) > jaccard_threshold:
                    is_duplicate = True
                    break

        if is_duplicate:
            reason = f"Duplicate or near-duplicate text (Jaccard > {jaccard_threshold})"
            rejected.append({**cand, "rejection_reason": reason})
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        # Passed all gates
        accepted.append(cand)

    telemetry = {
        "examples_generated": len(candidate_examples),
        "examples_accepted": len(accepted),
        "examples_rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / len(candidate_examples), 4) if candidate_examples else 0.0,
        "rejection_reasons": rejection_reasons
    }

    return {
        "accepted": accepted,
        "rejected": rejected,
        "telemetry": telemetry
    }
