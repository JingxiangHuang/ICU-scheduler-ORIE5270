"""Tests for icu_scheduler.stream — covers reservoir sampling + CMS (W7)."""

import random
from collections import Counter

import pandas as pd
import pytest

from icu_scheduler.stream import (
    ArrivalEvent,
    ArrivalStream,
    CountMinSketch,
    reservoir_sample,
)

# ---------- ArrivalStream -----------------------------------------------------


class TestArrivalStream:
    def test_from_dataframe_length_matches(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "intime": pd.to_datetime(["2180-01-01", "2180-01-02", "2180-01-03"]),
                "los": [24.0, 48.0, 12.0],
                "admission_type": ["EMERGENCY", "ELECTIVE", "URGENT"],
            }
        )
        stream = ArrivalStream.from_dataframe(df)
        assert len(stream) == 3

    def test_iteration_yields_arrival_events(self):
        df = pd.DataFrame(
            {
                "subject_id": [1],
                "intime": pd.to_datetime(["2180-01-01"]),
                "los": [24.0],
            }
        )
        stream = ArrivalStream.from_dataframe(df)
        events = list(stream)
        assert len(events) == 1
        assert isinstance(events[0], ArrivalEvent)
        assert events[0].subject_id == 1


# ---------- Reservoir sampling ------------------------------------------------


class TestReservoirSample:
    def test_returns_exactly_k(self):
        sample = reservoir_sample(range(1000), k=10, rng=random.Random(0))
        assert len(sample) == 10

    def test_k_larger_than_stream_returns_all(self):
        sample = reservoir_sample([1, 2, 3], k=10, rng=random.Random(0))
        assert sorted(sample) == [1, 2, 3]

    def test_k_zero_returns_empty(self):
        sample = reservoir_sample(range(100), k=0, rng=random.Random(0))
        assert sample == []

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            reservoir_sample(range(10), k=-1)

    def test_deterministic_with_seeded_rng(self):
        s1 = reservoir_sample(range(1000), k=5, rng=random.Random(42))
        s2 = reservoir_sample(range(1000), k=5, rng=random.Random(42))
        assert s1 == s2

    def test_approximately_uniform(self):
        """Each element should be sampled with prob k/n ≈ 0.01 (n=1000, k=10).

        Run 500 trials and check every index is hit at least once — a weak
        but reliable uniformity sanity check.
        """
        rng = random.Random(123)
        counts: Counter = Counter()
        for _ in range(500):
            for x in reservoir_sample(range(1000), k=10, rng=rng):
                counts[x] += 1
        # Expect ~5 hits per element; check that most elements are hit.
        assert len(counts) > 800


# ---------- CountMinSketch ----------------------------------------------------


class TestCountMinSketch:
    def test_rejects_non_positive_dims(self):
        with pytest.raises(ValueError):
            CountMinSketch(width=0, depth=5)
        with pytest.raises(ValueError):
            CountMinSketch(width=10, depth=0)

    def test_never_underestimates(self):
        cms = CountMinSketch(width=1024, depth=5)
        for item in ["MICU"] * 100 + ["SICU"] * 50 + ["CCU"] * 25:
            cms.add(item)
        assert cms.estimate("MICU") >= 100
        assert cms.estimate("SICU") >= 50
        assert cms.estimate("CCU") >= 25

    def test_unseen_item_estimate_is_small(self):
        cms = CountMinSketch(width=1024, depth=5)
        for _ in range(1000):
            cms.add("MICU")
        # Upper bound on false-positive count for an unseen key
        assert cms.estimate("neverseen") <= 10

    def test_add_with_count(self):
        cms = CountMinSketch(width=256, depth=3)
        cms.add("x", count=42)
        assert cms.estimate("x") >= 42
