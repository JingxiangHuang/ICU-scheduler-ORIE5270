"""Integer-programming optimizer for ICU admission allocation.

This module provides an offline optimal benchmark: given a known arrival stream
and each patient's length of stay, choose which admissions to accept so ICU
capacity is never exceeded while total clinical benefit is maximized.

The online policies in :mod:`icu_scheduler.scheduler` are heuristics. This ILP
is the "best possible with hindsight" comparison point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from icu_scheduler.scheduler import AdmissionDecision, AllocationPolicy, ICUState
from icu_scheduler.stream import ArrivalEvent, ArrivalStream


URGENT_ACUITIES = {"emergency", "urgent"}


class ArrivalForecaster(Protocol):
    """Protocol for forecast models used by rolling-horizon optimization."""

    def predict(
        self,
        now: pd.Timestamp,
        horizon_hours: float,
    ) -> List[ArrivalEvent]:
        """Return predicted future arrivals in the horizon."""


@dataclass(frozen=True)
class OptimizedAdmission:
    """One optimized admission/diversion decision."""

    subject_id: int
    arrival_time: pd.Timestamp
    discharge_time: pd.Timestamp
    acuity: str
    benefit: float
    admitted: bool


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by :func:`optimize_admissions`."""

    decisions: List[OptimizedAdmission]
    objective_value: float
    solver_status: int
    solver_message: str
    n_beds: int

    @property
    def n_admitted(self) -> int:
        return sum(d.admitted for d in self.decisions)

    @property
    def n_diverted(self) -> int:
        return len(self.decisions) - self.n_admitted

    @property
    def urgent_admit_rate(self) -> float:
        urgent = [d for d in self.decisions if d.acuity.lower() in URGENT_ACUITIES]
        if not urgent:
            return 0.0
        return sum(d.admitted for d in urgent) / len(urgent)

    @property
    def elective_admit_rate(self) -> float:
        elective = [d for d in self.decisions if d.acuity.lower() not in URGENT_ACUITIES]
        if not elective:
            return 0.0
        return sum(d.admitted for d in elective) / len(elective)

    def to_dataframe(self) -> pd.DataFrame:
        """Return decisions as a tidy DataFrame suitable for CSV export."""
        return pd.DataFrame(
            [
                {
                    "subject_id": d.subject_id,
                    "arrival_time": d.arrival_time,
                    "discharge_time": d.discharge_time,
                    "acuity": d.acuity,
                    "benefit": d.benefit,
                    "admitted": d.admitted,
                }
                for d in self.decisions
            ]
        )

    def occupancy_series(self) -> pd.Series:
        """Return optimized occupancy after each arrival timestamp."""
        if not self.decisions:
            return pd.Series(dtype=float)
        times = sorted({d.arrival_time for d in self.decisions})
        occupancy = []
        admitted = [d for d in self.decisions if d.admitted]
        for t in times:
            active = sum(d.arrival_time <= t < d.discharge_time for d in admitted)
            occupancy.append(active)
        return pd.Series(occupancy, index=pd.DatetimeIndex(times), name="occupancy")


def optimize_admissions(
    stream: Iterable[ArrivalEvent] | ArrivalStream,
    n_beds: int,
    urgent_benefit: float = 100.0,
    elective_benefit: float = 20.0,
    benefit_by_acuity: Optional[Mapping[str, float]] = None,
) -> OptimizationResult:
    """Solve the offline ICU admission optimization problem.

    The decision variable ``x_i`` is 1 if patient ``i`` is admitted and 0 if
    diverted. For every interval between an arrival/discharge event, the model
    enforces:

    ``sum(active_i_at_interval * x_i) <= n_beds``

    The objective maximizes total benefit. Urgent/emergency patients receive a
    higher default benefit, so the optimizer can explicitly trade elective
    cancellations against urgent protection.

    Parameters
    ----------
    stream
        Arrival events with ``arrival_time``, ``los_hours`` and ``acuity``.
    n_beds
        ICU bed capacity.
    urgent_benefit
        Benefit assigned to admitting emergency/urgent arrivals.
    elective_benefit
        Benefit assigned to admitting all other arrivals.
    benefit_by_acuity
        Optional mapping overriding benefits for specific acuity strings.

    Returns
    -------
    OptimizationResult
        Optimized admit/divert decisions and summary helpers.
    """
    if n_beds <= 0:
        raise ValueError("n_beds must be positive")
    if urgent_benefit < 0 or elective_benefit < 0:
        raise ValueError("benefits must be non-negative")

    events = list(stream)
    if not events:
        return OptimizationResult([], 0.0, 0, "empty input", n_beds=n_beds)

    starts, ends = _event_windows(events)
    benefits = np.asarray(
        [
            _benefit_for_event(
                event,
                urgent_benefit=urgent_benefit,
                elective_benefit=elective_benefit,
                benefit_by_acuity=benefit_by_acuity,
            )
            for event in events
        ],
        dtype=float,
    )

    admitted, objective_value, status, message = _solve_admission_ilp(
        starts=starts,
        ends=ends,
        benefits=benefits,
        n_beds=n_beds,
    )
    decisions = [
        OptimizedAdmission(
            subject_id=event.subject_id,
            arrival_time=event.arrival_time,
            discharge_time=event.arrival_time + pd.Timedelta(hours=float(event.los_hours)),
            acuity=event.acuity,
            benefit=float(benefits[i]),
            admitted=bool(admitted[i]),
        )
        for i, event in enumerate(events)
    ]

    return OptimizationResult(
        decisions=decisions,
        objective_value=objective_value,
        solver_status=status,
        solver_message=message,
        n_beds=n_beds,
    )


class RollingHorizonILPPolicy(AllocationPolicy):
    """Online optimization policy using a rolling finite-horizon ILP.

    At each arrival, the policy solves a small admission-selection ILP over the
    current patient plus predicted arrivals inside ``horizon_hours``. It then
    executes only the current patient's admit/divert decision and re-optimizes
    at the next arrival.

    This is deliberately different from :func:`optimize_admissions`: the
    offline optimizer sees the whole dataset, while this policy only sees the
    current rolling window and the current discharge schedule.
    """

    def __init__(
        self,
        n_beds: int,
        horizon_hours: float = 72.0,
        urgent_benefit: float = 100.0,
        elective_benefit: float = 20.0,
        max_board: int = 0,
        max_board_wait_hours: float = 24.0,
        forecaster: Optional[ArrivalForecaster] = None,
        allow_step_down: bool = True,
        max_transfer_remaining_hours: float = 24.0,
        transferable_acuities: tuple[str, ...] = ("elective", "observation"),
    ):
        if n_beds <= 0:
            raise ValueError("n_beds must be positive")
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if max_board < 0:
            raise ValueError("max_board must be non-negative")
        if max_board_wait_hours <= 0:
            raise ValueError("max_board_wait_hours must be positive")
        if max_transfer_remaining_hours <= 0:
            raise ValueError("max_transfer_remaining_hours must be positive")
        self.n_beds = n_beds
        self.horizon_hours = float(horizon_hours)
        self.urgent_benefit = float(urgent_benefit)
        self.elective_benefit = float(elective_benefit)
        self.max_board = max_board
        self.max_board_wait_hours = float(max_board_wait_hours)
        self.forecaster = forecaster
        self.allow_step_down = allow_step_down
        self.max_transfer_remaining_hours = float(max_transfer_remaining_hours)
        self.transferable_acuities = tuple(
            acuity.lower() for acuity in transferable_acuities
        )

    def decide(self, event: ArrivalEvent, state: ICUState) -> str:
        """Fallback used if a simulator does not provide rolling context."""
        if state.free > 0:
            return AdmissionDecision.ADMIT
        if state.board_queue_size < self.max_board:
            return AdmissionDecision.BOARD
        return AdmissionDecision.DIVERT

    def decide_with_context(
        self,
        event: ArrivalEvent,
        state: ICUState,
        future_events: Iterable[ArrivalEvent],
        active_discharge_offsets: Iterable[float],
        active_patients: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> str:
        """Optimize over the current rolling window and decide for ``event``."""
        release_offsets = list(active_discharge_offsets)
        window = self._candidate_window(event, future_events)
        if not window:
            return self.decide(event, state)

        starts = np.asarray(
            [
                (candidate.arrival_time - event.arrival_time).total_seconds() / 3600.0
                for candidate in window
            ],
            dtype=float,
        )
        ends = starts + np.asarray([candidate.los_hours for candidate in window], dtype=float)
        benefits = np.asarray(
            [
                _benefit_for_event(
                    candidate,
                    urgent_benefit=self.urgent_benefit,
                    elective_benefit=self.elective_benefit,
                    benefit_by_acuity=None,
                )
                for candidate in window
            ],
            dtype=float,
        )
        admitted, _objective, _status, _message = _solve_admission_ilp(
            starts=starts,
            ends=ends,
            benefits=benefits,
            n_beds=self.n_beds,
            existing_release_offsets=release_offsets,
        )

        if bool(admitted[0]):
            return AdmissionDecision.ADMIT
        if self._has_transfer_candidate(event, state, active_patients):
            return AdmissionDecision.TRANSFER
        usable_releases = sorted(offset for offset in release_offsets if offset > 0)
        queue_position = state.board_queue_size
        release_soon = (
            len(usable_releases) > queue_position
            and usable_releases[queue_position] <= self.max_board_wait_hours
        )
        if (
            state.free <= 0
            and state.board_queue_size < self.max_board
            and release_soon
        ):
            return AdmissionDecision.BOARD
        return AdmissionDecision.DIVERT

    def _has_transfer_candidate(
        self,
        event: ArrivalEvent,
        state: ICUState,
        active_patients: Optional[Iterable[Mapping[str, object]]],
    ) -> bool:
        """Return True when a conservative step-down transfer is available."""
        if not self.allow_step_down:
            return False
        if event.acuity.lower() not in URGENT_ACUITIES:
            return False
        if state.free > 0 or active_patients is None:
            return False

        allowed = set(self.transferable_acuities)
        for patient in active_patients:
            acuity = str(patient.get("acuity", "")).lower()
            if acuity not in allowed:
                continue
            try:
                remaining = float(patient.get("remaining_hours", float("inf")))
            except (TypeError, ValueError):
                continue
            if 0 <= remaining <= self.max_transfer_remaining_hours:
                return True
        return False

    def _candidate_window(
        self,
        event: ArrivalEvent,
        future_events: Iterable[ArrivalEvent],
    ) -> List[ArrivalEvent]:
        if self.forecaster is not None:
            return [event, *self.forecaster.predict(event.arrival_time, self.horizon_hours)]
        return self._perfect_window_events(event, future_events)

    def _perfect_window_events(
        self,
        event: ArrivalEvent,
        future_events: Iterable[ArrivalEvent],
    ) -> List[ArrivalEvent]:
        window = []
        for candidate in future_events:
            offset = (candidate.arrival_time - event.arrival_time).total_seconds() / 3600.0
            if offset < 0:
                continue
            if offset <= self.horizon_hours:
                window.append(candidate)
        return window


def _benefit_for_event(
    event: ArrivalEvent,
    urgent_benefit: float,
    elective_benefit: float,
    benefit_by_acuity: Optional[Mapping[str, float]],
) -> float:
    acuity = event.acuity.lower()
    if benefit_by_acuity and acuity in benefit_by_acuity:
        benefit = float(benefit_by_acuity[acuity])
    elif acuity in URGENT_ACUITIES:
        benefit = float(urgent_benefit)
    else:
        benefit = float(elective_benefit)
    if benefit < 0:
        raise ValueError("benefits must be non-negative")
    return benefit


def _solve_admission_ilp(
    starts: np.ndarray,
    ends: np.ndarray,
    benefits: np.ndarray,
    n_beds: int,
    existing_release_offsets: Optional[Iterable[float]] = None,
) -> tuple[np.ndarray, float, int, str]:
    constraints = _capacity_constraints(
        starts=starts,
        ends=ends,
        n_beds=n_beds,
        existing_release_offsets=existing_release_offsets,
    )
    result = milp(
        c=-benefits,
        integrality=np.ones(len(starts), dtype=int),
        bounds=Bounds(0, 1),
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"ILP solver failed: {result.message}")
    return (
        np.rint(result.x).astype(bool),
        float(-result.fun),
        int(result.status),
        str(result.message),
    )


def _event_windows(events: List[ArrivalEvent]) -> tuple[np.ndarray, np.ndarray]:
    base = min(event.arrival_time for event in events)
    starts = []
    ends = []
    for event in events:
        if event.los_hours < 0:
            raise ValueError("los_hours must be non-negative")
        start = (event.arrival_time - base).total_seconds() / 3600.0
        starts.append(start)
        ends.append(start + float(event.los_hours))
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _capacity_constraints(
    starts: np.ndarray,
    ends: np.ndarray,
    n_beds: int,
    existing_release_offsets: Optional[Iterable[float]] = None,
) -> Optional[LinearConstraint]:
    existing_iter = [] if existing_release_offsets is None else existing_release_offsets
    existing = np.asarray(
        [
            float(offset)
            for offset in existing_iter
            if float(offset) > 0
        ],
        dtype=float,
    )
    endpoints = np.unique(np.concatenate([starts, ends, np.asarray([0.0]), existing]))
    if len(endpoints) < 2:
        return None

    intervals = [
        (left + right) / 2.0
        for left, right in zip(endpoints[:-1], endpoints[1:])
        if right > left
    ]
    if not intervals:
        return None

    active = np.zeros((len(intervals), len(starts)), dtype=float)
    upper_bounds = np.zeros(len(intervals), dtype=float)
    for row, midpoint in enumerate(intervals):
        active[row, :] = (starts <= midpoint) & (midpoint < ends)
        occupied_existing = int(np.sum(existing > midpoint))
        upper_bounds[row] = max(float(n_beds - occupied_existing), 0.0)
    return LinearConstraint(
        active,
        lb=np.full(len(intervals), -np.inf),
        ub=upper_bounds,
    )
