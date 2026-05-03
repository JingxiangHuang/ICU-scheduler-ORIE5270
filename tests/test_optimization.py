"""Tests for the ILP-based offline optimizer."""

import pandas as pd
import pytest

from icu_scheduler.optimization import optimize_admissions
from icu_scheduler.optimization import RollingHorizonILPPolicy
from icu_scheduler.simulator import MonteCarloSimulator
from icu_scheduler.stream import ArrivalEvent, ArrivalStream


def _event(subject_id, hour, los_hours, acuity):
    return ArrivalEvent(
        subject_id=subject_id,
        arrival_time=pd.Timestamp("2180-01-01") + pd.Timedelta(hours=hour),
        los_hours=los_hours,
        acuity=acuity,
    )


class FixedForecaster:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, now, horizon_hours):
        return [
            ArrivalEvent(
                subject_id=event.subject_id,
                arrival_time=now + (event.arrival_time - pd.Timestamp("2180-01-01")),
                los_hours=event.los_hours,
                acuity=event.acuity,
            )
            for event in self.predictions
        ]


class TestOptimizeAdmissions:
    def test_prefers_urgent_when_capacity_conflicts(self):
        """With one bed, admitting the urgent patient is the optimal tradeoff."""
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10, acuity="elective"),
                _event(2, hour=1, los_hours=2, acuity="urgent"),
                _event(3, hour=11, los_hours=1, acuity="elective"),
            ]
        )

        result = optimize_admissions(stream, n_beds=1, urgent_benefit=10, elective_benefit=1)
        by_id = {d.subject_id: d for d in result.decisions}

        assert by_id[1].admitted is False
        assert by_id[2].admitted is True
        assert by_id[3].admitted is True
        assert result.objective_value == pytest.approx(11.0)
        assert result.urgent_admit_rate == 1.0

    def test_admits_non_overlapping_patients(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=1, acuity="elective"),
                _event(2, hour=1, los_hours=1, acuity="elective"),
            ]
        )

        result = optimize_admissions(stream, n_beds=1)

        assert result.n_admitted == 2
        assert result.n_diverted == 0

    def test_optimized_occupancy_never_exceeds_capacity(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=5, acuity="elective"),
                _event(2, hour=1, los_hours=5, acuity="urgent"),
                _event(3, hour=2, los_hours=5, acuity="emergency"),
                _event(4, hour=6, los_hours=2, acuity="elective"),
            ]
        )

        result = optimize_admissions(stream, n_beds=2)
        occupancy = result.occupancy_series()

        assert (occupancy <= 2).all()

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            optimize_admissions([], n_beds=0)
        with pytest.raises(ValueError):
            optimize_admissions([_event(1, 0, -1, "urgent")], n_beds=1)


class TestRollingHorizonILPPolicy:
    def test_forecast_driven_policy_does_not_need_true_future_events(self):
        stream = ArrivalStream([_event(1, hour=0, los_hours=10, acuity="elective")])
        forecaster = FixedForecaster([_event(99, hour=1, los_hours=2, acuity="urgent")])
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            urgent_benefit=10,
            elective_benefit=1,
            forecaster=forecaster,
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_admitted == 0
        assert result.n_diverted == 1

    def test_defers_elective_when_urgent_is_inside_horizon(self):
        """The online optimizer should save the only bed for the urgent case."""
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10, acuity="elective"),
                _event(2, hour=1, los_hours=2, acuity="urgent"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            urgent_benefit=10,
            elective_benefit=1,
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_admitted == 1
        assert result.n_diverted == 1
        assert result.n_urgent_admitted == 1
        assert result.n_elective_admitted == 0

    def test_short_horizon_behaves_myopically(self):
        """If the urgent arrival is outside the window, the elective is admitted."""
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10, acuity="elective"),
                _event(2, hour=5, los_hours=2, acuity="urgent"),
            ]
        )
        policy = RollingHorizonILPPolicy(n_beds=1, horizon_hours=1)

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_elective_admitted == 1

    def test_limited_boarding_when_bed_releases_soon(self):
        """A full ICU can board a high-priority arrival if a bed opens soon."""
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=1.0, acuity="elective"),
                _event(2, hour=0.5, los_hours=2.0, acuity="urgent"),
                _event(3, hour=1.1, los_hours=1.0, acuity="elective"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            max_board=1,
            forecaster=FixedForecaster([]),
            allow_step_down=False,
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_urgent_boarded == 1
        assert result.total_boarding_hours >= 0.5

    def test_diverts_when_boarding_wait_would_be_too_long(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10.0, acuity="elective"),
                _event(2, hour=1, los_hours=2.0, acuity="urgent"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            max_board=1,
            max_board_wait_hours=2,
            forecaster=FixedForecaster([]),
            allow_step_down=False,
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_urgent_boarded == 0
        assert result.n_urgent_diverted == 1

    def test_step_down_transfers_near_discharge_elective_for_urgent(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10.0, acuity="elective"),
                _event(2, hour=1, los_hours=2.0, acuity="urgent"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            max_board=0,
            max_transfer_remaining_hours=24,
            forecaster=FixedForecaster([]),
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_transferred == 1
        assert result.n_urgent_admitted == 1
        assert result.n_diverted == 0

    def test_step_down_respects_remaining_los_limit(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10.0, acuity="elective"),
                _event(2, hour=1, los_hours=2.0, acuity="urgent"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            max_board=0,
            max_transfer_remaining_hours=2,
            forecaster=FixedForecaster([]),
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_transferred == 0
        assert result.n_urgent_diverted == 1

    def test_step_down_only_for_urgent_current_arrivals(self):
        stream = ArrivalStream(
            [
                _event(1, hour=0, los_hours=10.0, acuity="elective"),
                _event(2, hour=1, los_hours=2.0, acuity="elective"),
            ]
        )
        policy = RollingHorizonILPPolicy(
            n_beds=1,
            horizon_hours=24,
            max_board=0,
            max_transfer_remaining_hours=24,
            forecaster=FixedForecaster([]),
        )

        result = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=1,
            n_workers=1,
            perturb=False,
        ).run_once(0)

        assert result.n_transferred == 0
        assert result.n_elective_diverted == 1
