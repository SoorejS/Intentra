"""
Intentra V2.1 - Curriculum Scheduler Module
Manages stage-based training data curriculum, archetype distribution policies,
difficulty progression, and provenance tracking across optimization cycles.

Curriculum Stages:
  Stage 1: Canonical Anchors (Unambiguous, core class definitions, high lexical separability)
  Stage 2: Controlled Variations (Syntactic paraphrases, diverse lengths, stylistic shifts)
  Stage 3: Boundary Disambiguation (Nuanced edge cases for diagnosed confusion pairs)
  Stage 4: Hard Negatives & Contrastive Cases (Lexically misleading cues, subtle context flips)
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

STAGE_ARCHETYPES = {
    1: "canonical_anchor",
    2: "controlled_variation",
    3: "boundary_case",
    4: "hard_negative"
}

ARCHETYPE_DIFFICULTY = {
    "canonical_anchor": "EASY",
    "controlled_variation": "EASY_MEDIUM",
    "boundary_case": "MEDIUM_HARD",
    "hard_negative": "HARD",
    "contrastive_pair": "HARD"
}


class CurriculumPolicy:
    """Defines distribution ratios and constraints for curriculum-aware generation."""
    def __init__(
        self,
        min_canonical_ratio: float = 0.35,
        max_hard_negative_ratio: float = 0.20,
        max_boundary_ratio: float = 0.30,
        variation_ratio: float = 0.15,
        stage_progression_threshold: float = 0.010, # Min F1 lift to advance stage
        enforce_provenance: bool = True
    ):
        self.min_canonical_ratio = min_canonical_ratio
        self.max_hard_negative_ratio = max_hard_negative_ratio
        self.max_boundary_ratio = max_boundary_ratio
        self.variation_ratio = variation_ratio
        self.stage_progression_threshold = stage_progression_threshold
        self.enforce_provenance = enforce_provenance

    def calculate_stage_quotas(self, total_count: int, current_stage: int = 1) -> Dict[str, int]:
        """
        Compute archetype counts based on current curriculum stage and total budget.
        Guarantees strong anchor coverage to prevent premature boundary collapse.
        """
        if total_count <= 0:
            return {}

        if current_stage == 1:
            # Stage 1: Heavy anchor grounding (70% canonical, 30% variation)
            canonical = int(np.round(total_count * 0.70))
            variation = total_count - canonical
            boundary = 0
            hard_neg = 0
        elif current_stage == 2:
            # Stage 2: Balanced anchor & variation with light boundary (50% anchor, 35% variation, 15% boundary)
            canonical = int(np.round(total_count * 0.50))
            boundary = int(np.round(total_count * 0.15))
            hard_neg = 0
            variation = total_count - canonical - boundary
        elif current_stage == 3:
            # Stage 3: Boundary disambiguation with preserved anchor floor (40% anchor, 20% variation, 30% boundary, 10% hard-neg)
            canonical = max(int(total_count * self.min_canonical_ratio), int(np.round(total_count * 0.40)))
            boundary = int(np.round(total_count * 0.30))
            hard_neg = int(np.round(total_count * 0.10))
            variation = total_count - canonical - boundary - hard_neg
        else:
            # Stage 4: Full curriculum with bounded hard negatives (35% anchor, 15% variation, 30% boundary, 20% hard-neg)
            canonical = max(int(total_count * self.min_canonical_ratio), int(np.round(total_count * 0.35)))
            hard_neg = min(int(total_count * self.max_hard_negative_ratio), int(np.round(total_count * 0.20)))
            boundary = min(int(total_count * self.max_boundary_ratio), int(np.round(total_count * 0.30)))
            variation = total_count - canonical - boundary - hard_neg

        # Ensure exact total
        quotas = {
            "canonical_anchor": max(1, canonical),
            "controlled_variation": max(0, variation),
            "boundary_case": max(0, boundary),
            "hard_negative": max(0, hard_neg)
        }
        
        current_sum = sum(quotas.values())
        if current_sum != total_count:
            quotas["canonical_anchor"] += (total_count - current_sum)

        return quotas


def determine_curriculum_stage(
    history_evaluations: List[Dict[str, Any]],
    current_f1: float = 0.0,
    confusion_severity: str = "LOW"
) -> Tuple[int, str]:
    """
    Dynamically select curriculum stage based on model maturity and failure mode.
    Returns: (stage_number, stage_rationale)
    """
    if not history_evaluations:
        return 1, "Initial Anchor Grounding (Stage 1): Establishing core canonical representations."

    eval_count = len(history_evaluations)
    latest_f1 = history_evaluations[-1].get("macro_f1", current_f1)

    if latest_f1 < 0.45 or eval_count == 1:
        return 2, "Controlled Variation (Stage 2): Anchors established, expanding lexical and syntactic breadth."
    elif latest_f1 < 0.65 or confusion_severity in ["MEDIUM", "HIGH"]:
        return 3, "Boundary Disambiguation (Stage 3): Resolving specific diagnosed confusion pairs."
    else:
        return 4, "Hard Negative & Contrastive Hardening (Stage 4): Fine-grained adversarial separation."


def build_curriculum_dataset(
    total_count: int,
    stage: int = 1,
    classes: Optional[List[str]] = None,
    seed: int = 42,
    diagnostics: Optional[Dict[str, Any]] = None,
    policy: Optional[CurriculumPolicy] = None
) -> List[Dict[str, Any]]:
    """
    Build a curriculum-structured dataset with complete provenance and genuine seed-dependent sampling.
    """
    rng = np.random.default_rng(seed)
    if policy is None:
        policy = CurriculumPolicy()

    if classes is None:
        classes = ["refund_request", "cancellation_request", "billing_inquiry", "technical_support", "general_feedback"]

    quotas = policy.calculate_stage_quotas(total_count, current_stage=stage)
    dataset = []

    # Import canonical/boundary seed pools from benchmark_suite if available
    from core.benchmark_suite import NAIVE_SEEDS, V1_BOUNDARY_SEEDS

    # 1. Generate Canonical Anchors
    n_canonical = quotas.get("canonical_anchor", 0)
    for i in range(n_canonical):
        cls = classes[i % len(classes)]
        pool = NAIVE_SEEDS.get(cls, [f"Standard canonical example for {cls}."])
        idx = int(rng.integers(0, len(pool)))
        tmpl = pool[idx]
        ref_id = int(rng.integers(1000, 9999))
        text = tmpl.format(ref_id) if "{}" in tmpl else f"{tmpl} (Ref #{ref_id})"
        dataset.append({
            "text": text,
            "label": cls,
            "type": "canonical",
            "generation_stage": 1,
            "archetype": "canonical_anchor",
            "target_class": cls,
            "confusion_pair": None,
            "generation_reason": f"Canonical anchor for {cls}",
            "difficulty": "EASY",
            "source_error_ids": []
        })

    # 2. Generate Controlled Variations
    n_var = quotas.get("controlled_variation", 0)
    variation_prefixes = [
        "Please note:", "Urgent message:", "Inquiry regarding my account:", "Kindly assist with:",
        "Hello support team,", "Good morning,", "Regarding recent ticket:", "Official request:",
        "Customer service inquiry:", "Priority attention requested:"
    ]
    for i in range(n_var):
        cls = classes[i % len(classes)]
        pool = NAIVE_SEEDS.get(cls, [f"Standard example for {cls}."])
        idx = int(rng.integers(0, len(pool)))
        base_text = pool[idx]
        prefix_idx = int(rng.integers(0, len(variation_prefixes)))
        prefix = variation_prefixes[prefix_idx]
        ref_id = int(rng.integers(1000, 9999))
        text = f"{prefix} {base_text.format(ref_id) if '{}' in base_text else base_text}"
        dataset.append({
            "text": text,
            "label": cls,
            "type": "canonical",
            "generation_stage": 2,
            "archetype": "controlled_variation",
            "target_class": cls,
            "confusion_pair": None,
            "generation_reason": f"Syntactic variation for {cls}",
            "difficulty": "EASY_MEDIUM",
            "source_error_ids": []
        })

    # 3. Generate Boundary Examples
    n_bnd = quotas.get("boundary_case", 0)
    primary_pair = diagnostics.get("primary_focus_pair") if diagnostics else None
    for i in range(n_bnd):
        if primary_pair and i % 2 == 0:
            cls = primary_pair[0] if (i // 2) % 2 == 0 else primary_pair[1]
            pair_tuple = list(primary_pair)
        else:
            cls = classes[i % len(classes)]
            pair_tuple = [cls, classes[(i + 1) % len(classes)]]

        pool = V1_BOUNDARY_SEEDS.get(cls, [f"Boundary case for {cls}."])
        idx = int(rng.integers(0, len(pool)))
        tmpl = pool[idx]
        ref_id = int(rng.integers(1000, 9999))
        text = tmpl.format(ref_id) if "{}" in tmpl else f"{tmpl} (Boundary Ref #{ref_id})"
        dataset.append({
            "text": text,
            "label": cls,
            "type": "boundary",
            "generation_stage": 3,
            "archetype": "boundary_case",
            "target_class": cls,
            "confusion_pair": pair_tuple,
            "generation_reason": f"Boundary disambiguation between {pair_tuple[0]} and {pair_tuple[1]}",
            "difficulty": "MEDIUM_HARD",
            "source_error_ids": []
        })

    # 4. Generate Hard Negatives & Contrastive Cases
    n_hn = quotas.get("hard_negative", 0)
    for i in range(n_hn):
        if primary_pair and i % 2 == 0:
            cls = primary_pair[0]
            other_cls = primary_pair[1]
        else:
            cls = classes[i % len(classes)]
            other_classes = [c for c in classes if c != cls]
            other_cls = other_classes[int(rng.integers(0, len(other_classes)))]

        ref_id = int(rng.integers(1000, 9999))

        # Diverse hard-negative templates per class
        hn_templates = {
            "refund_request": [
                f"My cancellation was confirmed last week for order #{ref_id}, but I am specifically writing to demand my refund credit.",
                f"Although tech support answered ticket #{ref_id}, the payment double-charged and I need an immediate reimbursement.",
                f"Please disregard the billing explanation on invoice #{ref_id}; I am formally requesting a full money-back return."
            ],
            "cancellation_request": [
                f"I do not need a refund or invoice adjustments, please strictly terminate and cancel this active account #{ref_id}.",
                f"Even though tech support resolved bug #{ref_id}, we have decided to decommission and end our subscription service.",
                f"Stop processing payments immediately and close out contract #{ref_id}; this is a definitive service cancellation."
            ],
            "billing_inquiry": [
                f"Are refund transaction reversals subject to merchant invoice processing fees on statement #{ref_id}?",
                f"Can you explain the unexpected surcharge line item on invoice #{ref_id}? I am not asking for a refund yet.",
                f"We received a cancellation receipt for ticket #{ref_id} but need the final tax accounting breakdown."
            ],
            "technical_support": [
                f"The invoice payment portal is failing with an unhandled 500 error code when downloading statement #{ref_id}.",
                f"I requested a refund earlier, but now the whole web dashboard is throwing HTTP 403 authorization faults on session #{ref_id}.",
                f"Our account cancellation form crashes on submit with a JavaScript runtime exception #{ref_id}."
            ],
            "general_feedback": [
                f"I had to cancel earlier due to technical bugs, but your support team provided wonderful assistance on ticket #{ref_id}.",
                f"After sorting out our billing dispute #{ref_id}, I wanted to share suggestions for improving your pricing transparency.",
                f"Great experience with the refund resolution on case #{ref_id}; here is my constructive quarterly feedback."
            ]
        }

        pool = hn_templates.get(cls, [f"Regarding {other_cls} issue #{ref_id}, the true request is for {cls}."])
        tmpl_idx = int(rng.integers(0, len(pool)))
        text = pool[tmpl_idx]

        dataset.append({
            "text": text,
            "label": cls,
            "type": "hard_negative",
            "generation_stage": 4,
            "archetype": "hard_negative",
            "target_class": cls,
            "confusion_pair": [cls, other_cls],
            "generation_reason": f"Hard negative containing '{other_cls}' cues for true intent '{cls}'",
            "difficulty": "HARD",
            "source_error_ids": []
        })

    return dataset
