"""Tests for hospital-facing decision support helpers."""

import pandas as pd
import pytest

from icu_scheduler.decision_support import recommendation_gaps, recommend_policies


def _row(policy, urgent, cost, boarding, utilization, divert=0.2):
    return {
        "rate_multiplier": 1.0,
        "n_beds": 4,
        "policy": policy,
        "urgent_admit_rate": urgent,
        "clinical_cost_per_arrival": cost,
        "mean_boarding_hours": boarding,
        "mean_utilization": utilization,
        "board_rate": 0.1,
        "divert_rate": divert,
        "elective_admit_rate": 0.7,
    }


class TestRecommendPolicies:
    def test_highest_urgent_admit_rate_wins(self):
        metrics = pd.DataFrame(
            [
                _row("fcfs", urgent=0.70, cost=1.0, boarding=1.0, utilization=0.9),
                _row("threshold_r1", urgent=0.90, cost=5.0, boarding=3.0, utilization=0.7),
            ]
        )

        recs = recommend_policies(metrics)

        assert recs.loc[0, "recommended_policy"] == "threshold_r1"

    def test_lower_clinical_cost_breaks_urgent_tie(self):
        metrics = pd.DataFrame(
            [
                _row("threshold_r1", urgent=0.90, cost=3.0, boarding=1.0, utilization=0.8),
                _row("adaptive_threshold", urgent=0.90, cost=2.0, boarding=9.0, utilization=0.6),
            ]
        )

        recs = recommend_policies(metrics)

        assert recs.loc[0, "recommended_policy"] == "adaptive_threshold"

    def test_lower_boarding_time_breaks_cost_tie(self):
        metrics = pd.DataFrame(
            [
                _row("threshold_r1", urgent=0.90, cost=2.0, boarding=4.0, utilization=0.9),
                _row("threshold_r2", urgent=0.90, cost=2.0, boarding=1.0, utilization=0.6),
            ]
        )

        recs = recommend_policies(metrics)

        assert recs.loc[0, "recommended_policy"] == "threshold_r2"

    def test_recommendation_gaps_compare_against_fcfs(self):
        metrics = pd.DataFrame(
            [
                _row("fcfs", urgent=0.60, cost=5.0, boarding=4.0, utilization=0.8, divert=0.30),
                _row(
                    "threshold_r1",
                    urgent=0.80,
                    cost=4.0,
                    boarding=2.0,
                    utilization=0.7,
                    divert=0.20,
                ),
            ]
        )
        recs = recommend_policies(metrics)

        gaps = recommendation_gaps(metrics, recs)

        assert gaps.loc[0, "recommended_policy"] == "threshold_r1"
        assert gaps.loc[0, "urgent_admit_gap"] == pytest.approx(0.20)
        assert gaps.loc[0, "divert_gap"] == pytest.approx(0.10)
