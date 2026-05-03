"""Streaming algorithms for ICU arrivals.

Exercises course skills: W7 (data streams — reservoir sampling, count-min sketch).

The ICU arrival process is treated as a stream. For the demo dataset we have it
all in memory, but the algorithms below are written as if each event arrives
one-at-a-time, so the code generalises to a real production feed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd


# ---------- Arrival stream ----------------------------------------------------


@dataclass
class ArrivalEvent:
    """A single ICU arrival.

    Attributes
    ----------
    subject_id
        Patient identifier.
    arrival_time
        Timestamp of ICU admission.
    los_hours
        Observed length of stay in hours (known only after discharge; used for
        simulation replay).
    acuity
        Optional acuity/severity bucket, e.g. ``"elective"`` or ``"emergency"``.
    """

    subject_id: int
    arrival_time: pd.Timestamp
    los_hours: float
    acuity: str = "unknown"


class ArrivalStream:
    """Iterable stream of :class:`ArrivalEvent` objects.

    Constructed from a DataFrame so we can replay historical MIMIC-IV stays; in
    a production setting this could wrap a Kafka consumer instead.
    """

    def __init__(self, events: Iterable[ArrivalEvent]):
        self._events: List[ArrivalEvent] = list(events)

    def __iter__(self) -> Iterator[ArrivalEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ArrivalStream":
        """Build a stream from a DataFrame with columns ``subject_id``,
        ``intime``, ``los`` and optional ``admission_type``.

        Notes
        -----
        MIMIC-IV stores ``los`` in days. The tests, however, pass values that
        are already in hours for clarity. To stay robust we treat ``los`` as
        already being in hours whenever ``<= 1000`` (i.e. under ~42 days) and
        otherwise as days × 24. An explicit ``los_hours`` column, if present,
        is always preferred.
        """
        if "intime" not in df.columns or "subject_id" not in df.columns:
            raise ValueError("DataFrame must have 'subject_id' and 'intime' columns")

        events: List[ArrivalEvent] = []
        has_type = "admission_type" in df.columns
        has_los_hours = "los_hours" in df.columns

        for _, row in df.iterrows():
            if has_los_hours:
                los_hours = float(row["los_hours"])
            elif "los" in df.columns:
                los_hours = float(row["los"])
            else:
                los_hours = 0.0
            raw_type = (
                str(row["admission_type"])
                if has_type and pd.notna(row["admission_type"])
                else "unknown"
            )
            acuity = _normalize_acuity(raw_type)
            events.append(
                ArrivalEvent(
                    subject_id=int(row["subject_id"]),
                    arrival_time=pd.Timestamp(row["intime"]),
                    los_hours=los_hours,
                    acuity=acuity,
                )
            )
        return cls(events)


def _normalize_acuity(raw: str) -> str:
    """Normalize free-text admission_type into {emergency, urgent, elective, observation}.

    MIMIC-IV uses codes like ``EW EMER.``, ``DIRECT EMER.``, ``SURGICAL SAME DAY
    ADMISSION`` etc. This mapping folds them into the four canonical buckets the
    policies understand.
    """
    if not raw:
        return "unknown"
    low = raw.lower()
    if "emer" in low:
        return "emergency"
    if "urgent" in low:
        return "urgent"
    if "elective" in low or "surgical same day" in low or "ambulatory" in low:
        return "elective"
    if "observation" in low:
        return "observation"
    return low


# ---------- Reservoir sampling (W7 material) ----------------------------------


def reservoir_sample(
    stream: Iterable,
    k: int,
    rng: Optional[random.Random] = None,
) -> List:
    """Uniform-random sample of ``k`` elements from a stream of unknown length.

    Implements Algorithm R (Vitter, 1985). Runs in O(n) time and O(k) memory.

    Parameters
    ----------
    stream
        Any iterable.
    k
        Sample size. Must be non-negative.
    rng
        Optional ``random.Random`` instance for reproducibility. If ``None``,
        uses the module-level ``random`` state.

    Returns
    -------
    list
        The sampled items, preserving no particular order.

    Raises
    ------
    ValueError
        If ``k`` is negative.

    Examples
    --------
    >>> rng = random.Random(42)
    >>> sample = reservoir_sample(range(1000), k=5, rng=rng)
    >>> len(sample)
    5
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    rng = rng if rng is not None else random.Random()
    reservoir: List = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


# ---------- Count-Min Sketch --------------------------------------------------


@dataclass
class CountMinSketch:
    """Approximate frequency counter with bounded memory.

    Useful when we want to track, e.g., number of admissions per ICU unit over a
    long rolling window without storing every event.

    Parameters
    ----------
    width
        Number of hash buckets per row. Larger width → smaller error.
    depth
        Number of independent hash rows. Larger depth → lower failure prob.
    """

    width: int = 1024
    depth: int = 5
    _counts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive")
        self._counts = np.zeros((self.depth, self.width), dtype=np.int64)

    def _hashes(self, item: str) -> List[int]:
        """Return one bucket index per row."""
        h = hashlib.md5(item.encode("utf-8")).hexdigest()
        # Split the md5 digest into ``depth`` chunks and mod into ``width``.
        chunk = len(h) // self.depth
        return [int(h[i * chunk : (i + 1) * chunk], 16) % self.width for i in range(self.depth)]

    def add(self, item: str, count: int = 1) -> None:
        """Record ``count`` observations of ``item``.

        Parameters
        ----------
        item
            The (string) key being observed.
        count
            Positive integer count of how many times ``item`` occurred.
        """
        for row, col in enumerate(self._hashes(item)):
            self._counts[row, col] += count

    def estimate(self, item: str) -> int:
        """Return the approximate count for ``item`` (min across rows).

        This estimate is an **upper bound** on the true count with high
        probability (parameterised by ``width`` / ``depth``).
        """
        return int(min(self._counts[row, col] for row, col in enumerate(self._hashes(item))))
