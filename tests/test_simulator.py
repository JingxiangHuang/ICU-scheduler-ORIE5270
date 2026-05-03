"""Tests for icu_scheduler.simulator.

Covers both single-run correctness and the multiprocessing branch.
"""

import numpy as np
import pandas as pd
import pytest

from icu_scheduler.scheduler import FCFSPolicy, ThresholdPolicy
from icu_scheduler.simulator import MonteCarloSimulator, SimulationResult
from icu_scheduler.stream import ArrivalEvent, ArrivalStream


def _tiny_stream(n: int = 10) -> ArrivalStream:
    """10 back-to-back arrivals, each LOS = 1 hour."""
    events = [
        ArrivalEvent(
            subject_id=i,
            arrival_time=pd.Timestamp("2180-01-01") + pd.Timedelta(hours=i),
            los_hours=1.0,
            acuity="emergency",
        )
        for i in range(n)
    ]
    return ArrivalStream(events)


class TestMonteCarloSimulatorValidation:
    def test_n_runs_must_be_positive(self):
        with pytest.raises(ValueError):
            MonteCarloSimulator(FCFSPolicy(), _tiny_stream(), n_runs=0, n_workers=1)

    def test_n_workers_must_be_positive(self):
        with pytest.raises(ValueError):
            MonteCarloSimulator(FCFSPolicy(), _tiny_stream(), n_runs=1, n_workers=0)


class TestSimulationResult:
    def test_mean_utilization(self):
        res = SimulationResult(
            n_admitted=10,
            n_boarded=0,
            n_diverted=0,
            max_occupancy=5,
            utilization_series=np.array([0.2, 0.4, 0.6, 0.8]),
        )
        assert res.mean_utilization == pytest.approx(0.5)


class TestSimulatorSequential:
    def test_run_once_admits_all_when_capacity_ample(self):
        sim = MonteCarloSimulator(
            policy=FCFSPolicy(max_board=0),
            stream=_tiny_stream(10),
            n_runs=1,
            n_workers=1,
            perturb=False,
        )
        result = sim.run_once(run_id=0)
        assert result.n_admitted == 10
        assert result.n_diverted == 0

    def test_run_returns_n_runs_results(self):
        sim = MonteCarloSimulator(
            policy=FCFSPolicy(max_board=0),
            stream=_tiny_stream(10),
            n_runs=3,
            n_workers=1,
            perturb=False,
        )
        results = sim.run()
        assert len(results) == 3

    def test_threshold_policy_diverts_electives(self):
        """With only 1 bed and an elective arrival, threshold=1 must divert."""
        elective_events = [
            ArrivalEvent(
                subject_id=i,
                arrival_time=pd.Timestamp("2180-01-01") + pd.Timedelta(hours=i),
                los_hours=10.0,  # long — will cause congestion
                acuity="elective",
            )
            for i in range(5)
        ]
        sim = MonteCarloSimulator(
            policy=ThresholdPolicy(n_beds=1, reserve=1),
            stream=ArrivalStream(elective_events),
            n_runs=1,
            n_workers=1,
            perturb=False,
        )
        result = sim.run_once(run_id=0)
        # With reserve=1 and n_beds=1, all electives must divert
        assert result.n_admitted == 0
        assert result.n_diverted == 5
        assert result.n_elective_diverted == 5

    def test_records_urgent_boarding_breakdown(self):
        events = [
            ArrivalEvent(
                subject_id=1,
                arrival_time=pd.Timestamp("2180-01-01"),
                los_hours=10.0,
                acuity="urgent",
            ),
            ArrivalEvent(
                subject_id=2,
                arrival_time=pd.Timestamp("2180-01-01 00:01"),
                los_hours=1.0,
                acuity="urgent",
            ),
        ]
        sim = MonteCarloSimulator(
            policy=FCFSPolicy(n_beds=1, max_board=1),
            stream=ArrivalStream(events),
            n_runs=1,
            n_workers=1,
            perturb=False,
        )

        result = sim.run_once(run_id=0)

        assert result.n_urgent_boarded == 1


class TestSimulatorParallel:
    @pytest.mark.slow
    def test_parallel_matches_sequential_in_expectation(self):
        """Parallel + sequential should produce the same aggregate admit count
        when perturbation is disabled."""
        seq = MonteCarloSimulator(
            FCFSPolicy(), _tiny_stream(), n_runs=4, n_workers=1, perturb=False
        ).run()
        par = MonteCarloSimulator(
            FCFSPolicy(), _tiny_stream(), n_runs=4, n_workers=2, perturb=False
        ).run()
        assert sum(r.n_admitted for r in seq) == sum(r.n_admitted for r in par)
