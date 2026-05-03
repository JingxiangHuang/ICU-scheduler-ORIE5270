"""Monte-Carlo simulator for ICU occupancy.

Exercises course skills: W8 (multiprocessing — Pool.map over independent runs),
W6 (priority queue event loop), W7 (consumes an ``ArrivalStream``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Deque, List, Optional, Tuple

import numpy as np

from icu_scheduler.scheduler import (
    AdmissionDecision,
    AllocationPolicy,
    DischargeHeap,
    ICUState,
)
from icu_scheduler.stream import ArrivalStream
from icu_scheduler.stream import ArrivalEvent


@dataclass
class SimulationResult:
    """Result of a single simulation run."""

    n_admitted: int
    n_boarded: int
    n_diverted: int
    max_occupancy: int
    utilization_series: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    n_urgent_arrivals: int = 0
    n_urgent_admitted: int = 0
    n_urgent_boarded: int = 0
    n_urgent_diverted: int = 0
    n_elective_arrivals: int = 0
    n_elective_admitted: int = 0
    n_elective_boarded: int = 0
    n_elective_diverted: int = 0
    total_boarding_hours: float = 0.0
    n_transferred: int = 0

    @property
    def mean_utilization(self) -> float:
        if self.utilization_series.size == 0:
            return 0.0
        return float(self.utilization_series.mean())

    @property
    def mean_boarding_hours(self) -> float:
        if self.n_boarded == 0:
            return 0.0
        return self.total_boarding_hours / self.n_boarded


class MonteCarloSimulator:
    """Run many independent simulations of the same policy.

    Parameters
    ----------
    policy
        Any :class:`AllocationPolicy`.
    stream
        Historical :class:`ArrivalStream` used as an empirical distribution.
    n_runs
        Number of Monte-Carlo replications.
    n_workers
        Processes for ``multiprocessing.Pool``. If ``1``, runs sequentially
        (useful for testing).
    perturb
        If True, bootstrap-resample the arrival stream each run for variance.
    """

    def __init__(
        self,
        policy: AllocationPolicy,
        stream: ArrivalStream,
        n_runs: int = 100,
        n_workers: int = 1,
        perturb: bool = True,
        seed: Optional[int] = None,
    ):
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if n_workers <= 0:
            raise ValueError("n_workers must be positive")
        self.policy = policy
        self.stream = stream
        self.n_runs = n_runs
        self.n_workers = n_workers
        self.perturb = perturb
        self.seed = seed
        # Default bed count if the policy doesn't carry its own.
        self._default_n_beds = getattr(policy, "n_beds", 20)
        self._default_max_board = getattr(policy, "max_board", 0)

    def run_once(self, run_id: int) -> SimulationResult:
        """Single-run discrete-event simulation.

        Algorithm (testable step-by-step):
            1. Build (or bootstrap) the arrival list.
            2. Initialise occupancy = 0, discharge heap empty.
            3. For each arrival in chronological order:
                 a. Advance simulation clock, pop due discharges.
                 b. Ask policy.decide(event, state).
                 c. If ADMIT: occupancy += 1; schedule discharge.
                    If BOARD: enqueue in boarding list.
                    If DIVERT: record.
            4. Return :class:`SimulationResult`.
        """
        rng = np.random.default_rng(None if self.seed is None else self.seed + run_id)
        events = list(self.stream)
        if not events:
            return SimulationResult(
                n_admitted=0,
                n_boarded=0,
                n_diverted=0,
                max_occupancy=0,
                utilization_series=np.zeros(0),
            )

        if self.perturb:
            idx = rng.integers(0, len(events), size=len(events))
            events = [events[int(i)] for i in idx]
            events.sort(key=lambda e: e.arrival_time)

        t0 = events[0].arrival_time
        n_beds = self._default_n_beds
        max_board = self._default_max_board

        heap = DischargeHeap()
        occupied = 0
        board_queue: Deque[Tuple[ArrivalEvent, float]] = deque()
        n_admitted = n_boarded = n_diverted = 0
        n_urgent_arrivals = n_urgent_admitted = 0
        n_urgent_boarded = n_urgent_diverted = 0
        n_elective_arrivals = n_elective_admitted = 0
        n_elective_boarded = n_elective_diverted = 0
        total_boarding_hours = 0.0
        n_transferred = 0
        max_occ = 0
        utilization: list[float] = []

        for event_index, ev in enumerate(events):
            now = (ev.arrival_time - t0).total_seconds() / 3600.0

            # Discharge anyone whose time has come, and fill newly opened beds
            # from the finite boarding queue at the actual discharge time.
            due = heap.pop_due(now)
            while due:
                for discharge in due:
                    occupied -= 1
                    if occupied < 0:  # defensive; should never trigger
                        occupied = 0
                    if board_queue:
                        boarded_event, board_start = board_queue.popleft()
                        admit_time = max(discharge.discharge_time, board_start)
                        occupied += 1
                        heap.push(
                            admit_time + boarded_event.los_hours,
                            boarded_event.subject_id,
                            boarded_event.acuity,
                            admit_time,
                        )
                        total_boarding_hours += max(admit_time - board_start, 0.0)
                due = heap.pop_due(now)

            state = ICUState(
                n_beds=n_beds,
                occupied=occupied,
                board_queue_size=len(board_queue),
            )
            if hasattr(self.policy, "decide_with_context"):
                active_offsets = [t - now for t in heap.discharge_times() if t > now]
                decision = self.policy.decide_with_context(
                    ev,
                    state,
                    future_events=events[event_index:],
                    active_discharge_offsets=active_offsets,
                    active_patients=heap.active_patients(now),
                )
            else:
                decision = self.policy.decide(ev, state)

            is_urgent = ev.acuity.lower() in ("emergency", "urgent")
            if is_urgent:
                n_urgent_arrivals += 1
            else:
                n_elective_arrivals += 1

            if decision == AdmissionDecision.ADMIT:
                occupied += 1
                heap.push(now + ev.los_hours, ev.subject_id, ev.acuity, now)
                n_admitted += 1
                if is_urgent:
                    n_urgent_admitted += 1
                else:
                    n_elective_admitted += 1
            elif decision == AdmissionDecision.TRANSFER:
                max_remaining = getattr(self.policy, "max_transfer_remaining_hours", 24.0)
                transferable = getattr(
                    self.policy,
                    "transferable_acuities",
                    ("elective", "observation"),
                )
                candidate = heap.pop_transfer_candidate(
                    now=now,
                    max_remaining_hours=max_remaining,
                    transferable_acuities=transferable,
                )
                if candidate is not None:
                    n_transferred += 1
                    occupied = max(occupied - 1, 0)
                    occupied += 1
                    heap.push(now + ev.los_hours, ev.subject_id, ev.acuity, now)
                    n_admitted += 1
                    if is_urgent:
                        n_urgent_admitted += 1
                    else:
                        n_elective_admitted += 1
                elif len(board_queue) < max_board:
                    board_queue.append((ev, now))
                    n_boarded += 1
                    if is_urgent:
                        n_urgent_boarded += 1
                    else:
                        n_elective_boarded += 1
                else:
                    n_diverted += 1
                    if is_urgent:
                        n_urgent_diverted += 1
                    else:
                        n_elective_diverted += 1
            elif decision == AdmissionDecision.BOARD and len(board_queue) < max_board:
                board_queue.append((ev, now))
                n_boarded += 1
                if is_urgent:
                    n_urgent_boarded += 1
                else:
                    n_elective_boarded += 1
            else:
                n_diverted += 1
                if is_urgent:
                    n_urgent_diverted += 1
                else:
                    n_elective_diverted += 1

            max_occ = max(max_occ, occupied)
            utilization.append(occupied / n_beds if n_beds > 0 else 0.0)

        return SimulationResult(
            n_admitted=n_admitted,
            n_boarded=n_boarded,
            n_diverted=n_diverted,
            max_occupancy=max_occ,
            n_urgent_arrivals=n_urgent_arrivals,
            n_urgent_admitted=n_urgent_admitted,
            n_urgent_boarded=n_urgent_boarded,
            n_urgent_diverted=n_urgent_diverted,
            n_elective_arrivals=n_elective_arrivals,
            n_elective_admitted=n_elective_admitted,
            n_elective_boarded=n_elective_boarded,
            n_elective_diverted=n_elective_diverted,
            total_boarding_hours=total_boarding_hours,
            n_transferred=n_transferred,
            utilization_series=np.asarray(utilization, dtype=float),
        )

    def run(self) -> List[SimulationResult]:
        """Run ``n_runs`` simulations; parallel if ``n_workers > 1``."""
        if self.n_workers == 1:
            return [self.run_once(i) for i in range(self.n_runs)]

        with Pool(processes=self.n_workers) as pool:
            return pool.map(self.run_once, range(self.n_runs))
