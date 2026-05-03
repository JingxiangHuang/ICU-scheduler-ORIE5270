"""Allocation policies for ICU bed scheduling.

Exercises course skills: W4 (TDD — each policy has pure-function ``decide``
method that's trivial to unit-test) and W6 (priority queue for the queueing
model lives here too).
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from icu_scheduler.stream import ArrivalEvent


class AdmissionDecision:
    """Outcome of a scheduling decision."""

    ADMIT = "admit"
    BOARD = "board"  # Hold in ED / ward while waiting for a bed
    DIVERT = "divert"  # Send to another hospital (or cancel elective)
    TRANSFER = "transfer"  # Step down a current ICU patient, then admit


@dataclass
class ICUState:
    """Snapshot of current ICU state passed to each policy decision."""

    n_beds: int
    occupied: int
    board_queue_size: int = 0

    @property
    def free(self) -> int:
        return self.n_beds - self.occupied


class AllocationPolicy(ABC):
    """Base class for scheduling policies. Override :meth:`decide`."""

    @abstractmethod
    def decide(self, event: ArrivalEvent, state: ICUState) -> str:
        """Return one of ``AdmissionDecision.{ADMIT, BOARD, DIVERT}``."""
        raise NotImplementedError


class FCFSPolicy(AllocationPolicy):
    """First-come-first-served with no reservation.

    Admits whenever a bed is free; otherwise boards up to ``max_board``.

    Parameters
    ----------
    max_board
        Maximum boarding queue size. Arrivals beyond this are diverted.
    n_beds
        Number of ICU beds. Exposed so the simulator can read it consistently
        across all policy types. Defaults to ``20``.
    """

    def __init__(self, max_board: int = 5, n_beds: int = 20):
        if max_board < 0:
            raise ValueError("max_board must be non-negative")
        if n_beds <= 0:
            raise ValueError("n_beds must be positive")
        self.max_board = max_board
        self.n_beds = n_beds

    def decide(self, event: ArrivalEvent, state: ICUState) -> str:
        """Admit if a bed is free; else board up to ``max_board``; else divert."""
        if state.free > 0:
            return AdmissionDecision.ADMIT
        if state.board_queue_size < self.max_board:
            return AdmissionDecision.BOARD
        return AdmissionDecision.DIVERT


class ThresholdPolicy(AllocationPolicy):
    """Reserve ``reserve`` beds for emergencies.

    Elective admissions are diverted once free beds drop to ``reserve``;
    emergencies admit as long as any bed is free.
    """

    def __init__(self, n_beds: int, reserve: int = 2, max_board: int = 5):
        if reserve < 0 or reserve > n_beds:
            raise ValueError("reserve must be in [0, n_beds]")
        if max_board < 0:
            raise ValueError("max_board must be non-negative")
        self.n_beds = n_beds
        self.reserve = reserve
        self.max_board = max_board

    def decide(self, event: ArrivalEvent, state: ICUState) -> str:
        """Decide admission using the emergency-reservation rule.

        - If ICU is full: board (if queue has room), else divert.
        - Emergencies/urgent admit as long as any bed is free.
        - Electives admit only when ``free > reserve``, else divert.
        """
        is_urgent = event.acuity.lower() in ("emergency", "urgent")

        if state.free == 0:
            if state.board_queue_size < self.max_board:
                return AdmissionDecision.BOARD
            return AdmissionDecision.DIVERT

        if is_urgent:
            return AdmissionDecision.ADMIT

        # Elective: only admit if we leave more than ``reserve`` free beds.
        if state.free > self.reserve:
            return AdmissionDecision.ADMIT
        return AdmissionDecision.DIVERT


class AdaptiveThresholdPolicy(AllocationPolicy):
    """Dynamically reserve beds based on current ICU congestion.

    The reserve is 0 below 70% occupancy, 1 at 70-90% occupancy, and 2 at
    90% occupancy or above. Urgent arrivals still admit whenever any bed is
    free; the adaptive reserve only restricts elective arrivals.
    """

    def __init__(self, n_beds: int, max_board: int = 5):
        if n_beds <= 0:
            raise ValueError("n_beds must be positive")
        if max_board < 0:
            raise ValueError("max_board must be non-negative")
        self.n_beds = n_beds
        self.max_board = max_board

    def current_reserve(self, state: ICUState) -> int:
        """Return the reserve level implied by the current occupancy."""
        occ = state.occupied / state.n_beds
        if occ >= 0.90:
            return 2
        if occ >= 0.70:
            return 1
        return 0

    def decide(self, event: ArrivalEvent, state: ICUState) -> str:
        """Decide admission using the congestion-adaptive reserve rule."""
        is_urgent = event.acuity.lower() in ("emergency", "urgent")
        reserve = self.current_reserve(state)

        if state.free == 0:
            if state.board_queue_size < self.max_board:
                return AdmissionDecision.BOARD
            return AdmissionDecision.DIVERT

        if is_urgent:
            return AdmissionDecision.ADMIT

        if state.free > reserve:
            return AdmissionDecision.ADMIT
        return AdmissionDecision.DIVERT


# ---------- Priority queue for the discharge event loop -----------------------


@dataclass(order=True)
class _DischargeEvent:
    """Internal event used by the simulator's heap."""

    discharge_time: float
    subject_id: int = field(compare=False)
    acuity: str = field(default="unknown", compare=False)
    admit_time: float = field(default=0.0, compare=False)


class DischargeHeap:
    """Thin wrapper around ``heapq`` for scheduled discharge events.

    Exists primarily so the simulator has a clean, testable abstraction rather
    than mutating a raw list.
    """

    def __init__(self) -> None:
        self._heap: List[_DischargeEvent] = []

    def push(
        self,
        discharge_time: float,
        subject_id: int,
        acuity: str = "unknown",
        admit_time: float = 0.0,
    ) -> None:
        heapq.heappush(
            self._heap,
            _DischargeEvent(discharge_time, subject_id, acuity, admit_time),
        )

    def pop_due(self, now: float) -> List[_DischargeEvent]:
        """Remove and return all events with ``discharge_time <= now``.

        Events are returned in ascending order of ``discharge_time``.
        """
        due: List[_DischargeEvent] = []
        while self._heap and self._heap[0].discharge_time <= now:
            due.append(heapq.heappop(self._heap))
        return due

    def discharge_times(self) -> List[float]:
        """Return scheduled discharge times currently in the heap."""
        return [event.discharge_time for event in self._heap]

    def active_patients(self, now: float) -> List[dict]:
        """Return active ICU patients with remaining LOS metadata."""
        return [
            {
                "subject_id": event.subject_id,
                "acuity": event.acuity,
                "admit_time": event.admit_time,
                "discharge_time": event.discharge_time,
                "remaining_hours": max(event.discharge_time - now, 0.0),
            }
            for event in self._heap
            if event.discharge_time > now
        ]

    def pop_transfer_candidate(
        self,
        now: float,
        max_remaining_hours: float,
        transferable_acuities: tuple[str, ...] = ("elective", "observation"),
    ) -> _DischargeEvent | None:
        """Remove and return the safest step-down candidate, if one exists."""
        allowed = {acuity.lower() for acuity in transferable_acuities}
        best_idx = None
        best_remaining = None
        for idx, event in enumerate(self._heap):
            remaining = event.discharge_time - now
            if remaining < 0 or remaining > max_remaining_hours:
                continue
            if event.acuity.lower() not in allowed:
                continue
            if best_remaining is None or remaining < best_remaining:
                best_idx = idx
                best_remaining = remaining
        if best_idx is None:
            return None
        candidate = self._heap.pop(best_idx)
        heapq.heapify(self._heap)
        return candidate

    def __len__(self) -> int:
        return len(self._heap)
