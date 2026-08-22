"""
Tests for True Multi-Seed Independence in Dataset Generation and Benchmark Suites.
Verifies that:
1. Same seed produces 100% identical dataset content and fingerprint (Reproducibility).
2. Different seeds produce distinct dataset content and fingerprints (Independence).
3. Exact class balance and total sample budgets are strictly preserved across seeds.
"""

from collections import Counter
import pytest
from core.benchmark_suite import (
    build_naive_dataset,
    build_v1_dataset,
    build_v2_closed_loop_dataset,
    compute_dataset_fingerprint,
    get_dedicated_validation_set
)
from core.curriculum_scheduler import build_curriculum_dataset


def test_naive_dataset_seed_independence():
    """Verify build_naive_dataset produces identical data for same seed, distinct for different seeds."""
    count = 100
    seeds = [42, 123, 456, 789, 999]

    # 1. Reproducibility test
    d1_42 = build_naive_dataset(count, seed=42)
    d2_42 = build_naive_dataset(count, seed=42)
    fp1_42 = compute_dataset_fingerprint(d1_42)
    fp2_42 = compute_dataset_fingerprint(d2_42)
    assert fp1_42 == fp2_42, "Same seed must produce identical fingerprint"
    assert d1_42 == d2_42, "Same seed must produce identical dataset items"

    # 2. Independence & distinct fingerprints test
    fingerprints = set()
    for s in seeds:
        ds = build_naive_dataset(count, seed=s)
        assert len(ds) == count, f"Dataset size must be exactly {count}"
        fp = compute_dataset_fingerprint(ds)
        fingerprints.add(fp)

        # 3. Class balance verification
        counts = Counter(ex["label"] for ex in ds)
        assert len(counts) == 5, "All 5 classes must be represented"
        assert all(c == count // 5 for c in counts.values()), "Classes must be perfectly balanced"

    assert len(fingerprints) == len(seeds), f"Each seed must produce a unique fingerprint. Got {len(fingerprints)} / {len(seeds)}"


def test_v1_dataset_seed_independence():
    """Verify build_v1_dataset produces identical data for same seed, distinct for different seeds."""
    count = 100
    seeds = [42, 123, 456, 789, 999]

    d1_42 = build_v1_dataset(count, seed=42)
    d2_42 = build_v1_dataset(count, seed=42)
    assert compute_dataset_fingerprint(d1_42) == compute_dataset_fingerprint(d2_42)

    fingerprints = set()
    for s in seeds:
        ds = build_v1_dataset(count, seed=s)
        assert len(ds) == count
        fingerprints.add(compute_dataset_fingerprint(ds))

        counts = Counter(ex["label"] for ex in ds)
        assert len(counts) == 5
        assert all(c == count // 5 for c in counts.values())

    assert len(fingerprints) == len(seeds), "All seeds must produce unique V1 datasets"


def test_curriculum_dataset_seed_independence():
    """Verify build_curriculum_dataset produces identical data for same seed, distinct for different seeds."""
    count = 100
    seeds = [42, 123, 456, 789, 999]

    d1_42 = build_curriculum_dataset(count, stage=3, seed=42)
    d2_42 = build_curriculum_dataset(count, stage=3, seed=42)
    assert compute_dataset_fingerprint(d1_42) == compute_dataset_fingerprint(d2_42)

    fingerprints = set()
    for s in seeds:
        ds = build_curriculum_dataset(count, stage=3, seed=s)
        assert len(ds) == count
        fingerprints.add(compute_dataset_fingerprint(ds))

        counts = Counter(ex["label"] for ex in ds)
        assert len(counts) == 5
        assert all(c == count // 5 for c in counts.values())

    assert len(fingerprints) == len(seeds), "All seeds must produce unique curriculum datasets"


def test_v2_closed_loop_seed_independence():
    """Verify build_v2_closed_loop_dataset produces distinct fingerprints across seeds."""
    count = 100
    val_set = get_dedicated_validation_set()
    seeds = [42, 123, 456]

    d1_42 = build_v2_closed_loop_dataset(count, val_set=val_set, seed=42)
    d2_42 = build_v2_closed_loop_dataset(count, val_set=val_set, seed=42)
    assert d1_42["fingerprint"] == d2_42["fingerprint"]

    fingerprints = set()
    for s in seeds:
        obj = build_v2_closed_loop_dataset(count, val_set=val_set, seed=s)
        assert len(obj["dataset"]) == count
        fingerprints.add(obj["fingerprint"])

    assert len(fingerprints) == len(seeds), "All seeds must produce unique V2 datasets"
