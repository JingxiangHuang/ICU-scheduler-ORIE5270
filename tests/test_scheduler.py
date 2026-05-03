"""Tests for icu_scheduler.scheduler — policies + discharge heap.

Easy wins for coverage: policies are pure functions.
"""

import pandas as pd
import pytest

from icu_scheduler.scheduler import (
    AdmissionDecision,
    AdaptiveThresholdPolicy,
    DischargeHeap,
    FCFSPolicy,
    ICUState,
    ThresholdPolicy,
)
from icu_scheduler.stream import ArrivalEvent


def _event(acuity: str = "emergency") -> ArrivalEvent:
    return ArrivalEvent(
        subject_id=1,
        arrival_time=pd.Timestamp("2180-01-01 00:00"),
        los_hours=24.0,
        acuity=acuity,
    )


# ---------- ICUState ----------------------------------------------------------


class TestICUState:
    def test_free_beds_arithmetic(self):
        assert ICUState(n_beds=20, occupied=5).free == 15

    def test_free_beds_when_full(self):
        assert ICUState(n_beds=20, occupied=20).free == 0


# ---------- FCFS policy -------------------------------------------------------


class TestFCFSPolicy:
    def test_admits_when_free(self):
        p = FCFSPolicy(max_board=5)
        assert p.decide(_event(), ICUState(20, occupied=0)) == AdmissionDecision.ADMIT

    def test_boards_when_full_with_room_in_queue(self):
        p = FCFSPolicy(max_board=5)
        assert (
            p.decide(_event(), ICUState(20, occupied=20, board_queue_size=0))
            == AdmissionDecision.BOARD
        )

    def test_diverts_when_full_and_board_queue_full(self):
        p = FCFSPolicy(max_board=5)
        assert (
            p.decide(_event(), ICUState(20, occupied=20, board_queue_size=5))
            == AdmissionDecision.DIVERT
        )

    def test_negative_max_board_raises(self):
        with pytest.raises(ValueError):
            FCFSPolicy(max_board=-1)


# ---------- Threshold policy --------------------------------------------------


class TestThresholdPolicy:
    def test_invalid_reserve_raises(self):
        with pytest.raises(ValueError):
            ThresholdPolicy(n_beds=20, reserve=-1)
        with pytest.raises(ValueError):
            ThresholdPolicy(n_beds=20, reserve=25)

    def test_emergency_admits_when_any_bed_free(self):
        p = ThresholdPolicy(n_beds=20, reserve=2)
        assert p.decide(_event("emergency"), ICUState(20, occupied=19)) == AdmissionDecision.ADMIT

    def test_elective_diverted_within_reserve(self):
        p = ThresholdPolicy(n_beds=20, reserve=2)
        # 2 beds free = inside the reserve → elective diverts
        assert p.decide(_event("elective"), ICUState(20, occupied=18)) == AdmissionDecision.DIVERT

    def test_elective_admits_above_reserve(self):
        p = ThresholdPolicy(n_beds=20, reserve=2)
        assert p.decide(_event("elective"), ICUState(20, occupied=10)) == AdmissionDecision.ADMIT

    def test_emergency_boards_when_full(self):
        p = ThresholdPolicy(n_beds=20, reserve=2, max_board=3)
        assert (
            p.decide(_event("emergency"), ICUState(20, occupied=20, board_queue_size=1))
            == AdmissionDecision.BOARD
        )


# ---------- Adaptive threshold policy -----------------------------------------


class TestAdaptiveThresholdPolicy:
    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            AdaptiveThresholdPolicy(n_beds=0)
        with pytest.raises(ValueError):
            AdaptiveThresholdPolicy(n_beds=20, max_board=-1)

    def test_current_reserve_tracks_occupancy(self):
        p = AdaptiveThresholdPolicy(n_beds=10)

        assert p.current_reserve(ICUState(10, occupied=6)) == 0
        assert p.current_reserve(ICUState(10, occupied=7)) == 1
        assert p.current_reserve(ICUState(10, occupied=9)) == 2

    def test_emergency_admits_even_inside_adaptive_reserve(self):
        p = AdaptiveThresholdPolicy(n_beds=10)
        assert p.decide(_event("urgent"), ICUState(10, occupied=9)) == AdmissionDecision.ADMIT

    def test_elective_admits_when_free_beds_exceed_adaptive_reserve(self):
        p = AdaptiveThresholdPolicy(n_beds=10)
        assert p.decide(_event("elective"), ICUState(10, occupied=8)) == AdmissionDecision.ADMIT

    def test_elective_diverts_when_inside_adaptive_reserve(self):
        p = AdaptiveThresholdPolicy(n_beds=10)
        assert p.decide(_event("elective"), ICUState(10, occupied=9)) == AdmissionDecision.DIVERT

    def test_boards_then_diverts_when_full(self):
        p = AdaptiveThresholdPolicy(n_beds=10, max_board=2)

        assert (
            p.decide(_event("emergency"), ICUState(10, occupied=10, board_queue_size=1))
            == AdmissionDecision.BOARD
        )
        assert (
            p.decide(_event("emergency"), ICUState(10, occupied=10, board_queue_size=2))
            == AdmissionDecision.DIVERT
        )


# ---------- DischargeHeap -----------------------------------------------------


class TestDischargeHeap:
    def test_empty_initially(self):
        assert len(DischargeHeap()) == 0

    def test_push_increases_size(self):
        h = DischargeHeap()
        h.push(10.0, 1)
        h.push(5.0, 2)
        assert len(h) == 2

    def test_pop_due_returns_due_events_only(self):
        h = DischargeHeap()
        h.push(5.0, 1)
        h.push(10.0, 2)
        h.push(15.0, 3)
        due = h.pop_due(now=12.0)
        assert len(due) == 2
        assert {e.subject_id for e in due} == {1, 2}
        assert len(h) == 1  # stay_id=3 remains

    def test_pop_due_returns_in_time_order(self):
        h = DischargeHeap()
        h.push(10.0, 2)
        h.push(5.0, 1)
        h.push(15.0, 3)
        due = h.pop_due(now=100.0)
        times = [e.discharge_time for e in due]
        assert times == sorted(times)

    def test_active_patients_exposes_remaining_los(self):
        h = DischargeHeap()
        h.push(10.0, 1, acuity="elective", admit_time=0.0)
        h.push(3.0, 2, acuity="urgent", admit_time=0.0)

        active = h.active_patients(now=4.0)

        assert len(active) == 1
        assert active[0]["subject_id"] == 1
        assert active[0]["remaining_hours"] == pytest.approx(6.0)

    def test_pop_transfer_candidate_chooses_low_acuity_near_discharge(self):
        h = DischargeHeap()
        h.push(12.0, 1, acuity="elective", admit_time=0.0)
        h.push(6.0, 2, acuity="observation", admit_time=0.0)
        h.push(5.0, 3, acuity="urgent", admit_time=0.0)

        candidate = h.pop_transfer_candidate(now=4.0, max_remaining_hours=8.0)

        assert candidate is not None
        assert candidate.subject_id == 2
        assert {patient["subject_id"] for patient in h.active_patients(4.0)} == {1, 3}
