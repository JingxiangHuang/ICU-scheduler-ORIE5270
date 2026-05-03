"""Tests for icu_scheduler.mapreduce — word-count and domain aggregator."""

import pandas as pd
import pytest

from icu_scheduler.mapreduce import aggregate_careunit_hours, map_reduce


def _word_mapper(word):
    yield word, 1


def _sum_reducer(key, values):
    return sum(values)


class TestMapReduce:
    def test_word_count(self):
        out = map_reduce(["a", "b", "a", "a", "b", "c"], _word_mapper, _sum_reducer)
        assert out == {"a": 3, "b": 2, "c": 1}

    def test_empty_input(self):
        out = map_reduce([], _word_mapper, _sum_reducer)
        assert out == {}

    def test_negative_workers_raises(self):
        with pytest.raises(ValueError):
            map_reduce(["a"], _word_mapper, _sum_reducer, n_workers=0)

    def test_parallel_matches_sequential(self):
        data = ["a", "b", "a", "c", "b", "a"]
        seq = map_reduce(data, _word_mapper, _sum_reducer, n_workers=1)
        par = map_reduce(data, _word_mapper, _sum_reducer, n_workers=2)
        assert seq == par


class TestAggregateCareunitHours:
    def test_basic_aggregation(self):
        cohort = pd.DataFrame(
            {
                "first_careunit": ["MICU", "SICU", "MICU", "CCU"],
                "los": [1.0, 2.0, 0.5, 3.0],
            }
        )
        out = aggregate_careunit_hours(cohort)
        assert out["MICU"] == pytest.approx(1.5 * 24)
        assert out["SICU"] == pytest.approx(2.0 * 24)
        assert out["CCU"] == pytest.approx(3.0 * 24)

    def test_empty_cohort(self):
        cohort = pd.DataFrame({"first_careunit": [], "los": []})
        out = aggregate_careunit_hours(cohort)
        assert out == {}
