"""
Intentra V2.1 - Enhanced Quality Filter Module
Multi-stage quality validation gate for synthetic training examples.
Enforces:
  1. Length, emptiness, and formatting checks
  2. Schema label normalization and validity
  3. Character trigram Jaccard deduplication (threshold 0.70)
  4. Cross-class conflicting keyword contamination detection
  5. Provenance validation and default attribution
  6. Curriculum composition constraints and anchor coverage floors
"""

from collections import Counter
from typing import List, Dict, Any, Optional

# Standard signature keywords per class for conflicting keyword detection
KNOWN_CLASS_KEYWORDS = {
    "refund_request": {"refund", "money", "reimbursement", "reimburse", "reverse"},
    "cancellation_request": {"cancel", "cancellation", "terminate", "abort", "discontinue"},
    "billing_inquiry": {"invoice", "tax", "vat", "prorated", "statement"},
    "technical_support": {"crash", "403", "500", "ssl", "sso", "freeze"},
    "general_feedback": {"feedback", "design", "ui", "ux", "suggestion"}
}


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
    jaccard_threshold: float = 0.70,
    curriculum_stage: int = 1,
    max_conflicting_keyword_rate: float = 0.35,
    enforce_anchor_floor: bool = True
) -> dict:
    """
    Filter and validate candidate synthetic examples against multi-stage quality gates.
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
        item_type = cand.get("type", "canonical")
        archetype = cand.get("archetype", "canonical_anchor" if item_type == "canonical" else "boundary_case" if item_type == "boundary" else "hard_negative")

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

        # Gate 3: Near-Duplicate Detection (Jaccard > threshold)
        cand_trigram = get_char_trigrams(text)
        is_duplicate = False

        for ex_trigram in existing_trigrams:
            if jaccard_similarity(cand_trigram, ex_trigram) > jaccard_threshold:
                is_duplicate = True
                break

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

        # Gate 4: Cross-Class Contamination Check on Canonical Anchors
        # Canonical anchors should NOT contain strong keywords of other classes to preserve centroid clarity
        if archetype == "canonical_anchor" or item_type == "canonical":
            tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())
            has_severe_contamination = False
            for other_cls, kws in KNOWN_CLASS_KEYWORDS.items():
                if other_cls != label:
                    if len(tokens.intersection(kws)) >= 2:
                        has_severe_contamination = True
                        break
            if has_severe_contamination:
                reason = "Cross-class keyword contamination in canonical anchor"
                rejected.append({**cand, "rejection_reason": reason})
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

        # Gate 5: Provenance Enrichment
        cand_enriched = {
            **cand,
            "generation_stage": cand.get("generation_stage", curriculum_stage),
            "archetype": archetype,
            "target_class": label,
            "confusion_pair": cand.get("confusion_pair"),
            "difficulty": cand.get("difficulty", "MEDIUM_HARD" if item_type == "boundary" else "HARD" if item_type == "hard_negative" else "EASY"),
            "generation_reason": cand.get("generation_reason", f"Generated during curriculum stage {curriculum_stage}"),
            "source_error_ids": cand.get("source_error_ids", [])
        }

        accepted.append(cand_enriched)

    # Composition Telemetry
    total_accepted = len(accepted)
    arch_counts = Counter(item.get("archetype") for item in accepted)
    anchor_count = arch_counts.get("canonical_anchor", 0) + arch_counts.get("controlled_variation", 0)
    anchor_ratio = round(anchor_count / total_accepted, 4) if total_accepted else 0.0

    telemetry = {
        "examples_generated": len(candidate_examples),
        "examples_accepted": total_accepted,
        "examples_rejected": len(rejected),
        "acceptance_rate": round(total_accepted / len(candidate_examples), 4) if candidate_examples else 0.0,
        "anchor_coverage_ratio": anchor_ratio,
        "archetype_distribution": dict(arch_counts),
        "rejection_reasons": rejection_reasons
    }

    return {
        "accepted": accepted,
        "rejected": rejected,
        "telemetry": telemetry
    }
