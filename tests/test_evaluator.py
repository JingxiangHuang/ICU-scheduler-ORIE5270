"""Tests for icu_scheduler.evaluator — pure functions, easiest wins."""

import numpy as np
import pytest

from icu_scheduler.evaluator import (
    ClinicalCostWeights,
    KPISummary,
    compare_policies,
    compute_kpis,
    simulation_cost,
)
from icu_scheduler.simulator import SimulationResult


def _mk_result(
    admit=10,
    board=1,
    divert=2,
    max_occ=5,
    util=None,
    transferred=0,
) -> SimulationResult:
    if util is None:
        util = np.linspace(0.1, 0.9, 24)
    return SimulationResult(
        n_admitted=admit,
        n_boarded=board,
        n_diverted=divert,
        max_occupancy=max_occ,
        utilization_series=util,
        n_transferred=transferred,
    )


class TestComputeKpis:
    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            compute_kpis([])

    def test_rates_sum_to_one(self):
        results = [_mk_result(admit=10, board=1, divert=2) for _ in range(3)]
        kpi = compute_kpis(results)
        total = kpi.admit_rate + kpi.board_rate + kpi.divert_rate
        assert total == pytest.approx(1.0)

    def test_returns_kpi_summary(self):
        results = [_mk_result() for _ in range(5)]
        kpi = compute_kpis(results)
        assert isinstance(kpi, KPISummary)

    def test_p95_ge_mean(self):
        results = [_mk_result() for _ in range(5)]
        kpi = compute_kpis(results)
        assert kpi.p95_utilization >= kpi.mean_utilization

    def test_all_diverted_gives_zero_admit_rate(self):
        results = [_mk_result(admit=0, board=0, divert=10)]
        kpi = compute_kpis(results)
        assert kpi.admit_rate == 0.0
        assert kpi.divert_rate == 1.0

    def test_clinical_cost_penalizes_urgent_diversion_most(self):
        urgent_divert = SimulationResult(
            n_admitted=0,
            n_boarded=0,
            n_diverted=1,
            max_occupancy=0,
            n_urgent_arrivals=1,
            n_urgent_diverted=1,
        )
        elective_divert = SimulationResult(
            n_admitted=0,
            n_boarded=0,
            n_diverted=1,
            max_occupancy=0,
            n_elective_arrivals=1,
            n_elective_diverted=1,
        )

        assert simulation_cost(urgent_divert) > simulation_cost(elective_divert)

    def test_custom_cost_weights_flow_into_kpis(self):
        result = SimulationResult(
            n_admitted=0,
            n_boarded=1,
            n_diverted=0,
            max_occupancy=0,
            n_urgent_arrivals=1,
            n_urgent_boarded=1,
        )

        kpi = compute_kpis(
            [result],
            weights=ClinicalCostWeights(urgent_boarded=7, idle_bed_step=0),
        )

        assert kpi.clinical_cost_per_arrival == pytest.approx(7)

    def test_transfer_cost_and_rate_are_reported(self):
        result = _mk_result(admit=2, board=0, divert=0, transferred=1)

        kpi = compute_kpis(
            [result],
            weights=ClinicalCostWeights(transfer=4, idle_bed_step=0),
        )

        assert kpi.transfer_rate == pytest.approx(0.5)
        assert kpi.clinical_cost_per_arrival == pytest.approx(2.0)


class TestComparePolicies:
    def test_returns_row_per_policy(self):
        named = {
            "fcfs": [_mk_result(admit=10, board=1, divert=0)],
            "threshold": [_mk_result(admit=8, board=0, divert=3)],
        }
        df = compare_policies(named)
        assert len(df) == 2
        assert set(df["policy"]) == {"fcfs", "threshold"}
        assert "clinical_cost_per_arrival" in df.columns
        assert "mean_boarding_hours" in df.columns
        assert "transfer_rate" in df.columns
