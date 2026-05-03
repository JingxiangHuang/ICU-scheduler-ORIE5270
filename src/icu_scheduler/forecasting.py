"""Forecasting models for deployable rolling-horizon ICU optimization.

The optimizer should not see the true future arrival stream. This module
provides a small, auditable historical forecaster that can be trained on prior
arrivals and then used to generate a finite-horizon scenario for the ILP.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from icu_scheduler.stream import ArrivalEvent, ArrivalStream


@dataclass(frozen=True)
class HistoricalArrivalForecaster:
    """Deterministic arrival/LOS forecaster fitted from historical events.

    The model estimates:
    - arrivals per hour by acuity bucket
    - median LOS by acuity bucket

    At decision time it converts expected arrivals in the horizon into a
    deterministic scenario with evenly spaced predicted events. This is simple
    enough to deploy and audit, but it avoids leaking the exact future stream
    into the rolling ILP policy.
    """

    rates_per_hour: Dict[str, float]
    los_hours_by_acuity: Dict[str, float]
    default_los_hours: float
    max_events_per_acuity: int = 50

    @classmethod
    def from_events(
        cls,
        events: Iterable[ArrivalEvent],
        max_events_per_acuity: int = 50,
    ) -> "HistoricalArrivalForecaster":
        """Fit acuity-level rates and LOS medians from historical events."""
        event_list = sorted(list(events), key=lambda event: event.arrival_time)
        if not event_list:
            return cls({}, {}, default_los_hours=24.0)

        span_hours = max(
            (
                event_list[-1].arrival_time - event_list[0].arrival_time
            ).total_seconds()
            / 3600.0,
            1.0,
        )
        counts: Dict[str, int] = defaultdict(int)
        los_values: Dict[str, List[float]] = defaultdict(list)
        for event in event_list:
            acuity = event.acuity.lower()
            counts[acuity] += 1
            if event.los_hours > 0:
                los_values[acuity].append(float(event.los_hours))

        all_los = [event.los_hours for event in event_list if event.los_hours > 0]
        default_los = float(np.median(all_los)) if all_los else 24.0
        return cls(
            rates_per_hour={
                acuity: count / span_hours for acuity, count in counts.items()
            },
            los_hours_by_acuity={
                acuity: float(np.median(values)) if values else default_los
                for acuity, values in los_values.items()
            },
            default_los_hours=default_los,
            max_events_per_acuity=max_events_per_acuity,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        max_events_per_acuity: int = 50,
    ) -> "HistoricalArrivalForecaster":
        """Fit the forecaster from a MIMIC-shaped cohort DataFrame."""
        return cls.from_events(
            ArrivalStream.from_dataframe(df),
            max_events_per_acuity=max_events_per_acuity,
        )

    def predict(
        self,
        now: pd.Timestamp,
        horizon_hours: float,
    ) -> List[ArrivalEvent]:
        """Generate predicted future arrivals inside the horizon."""
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")

        predictions: List[ArrivalEvent] = []
        next_subject_id = -1
        for acuity, rate in sorted(self.rates_per_hour.items()):
            if rate <= 0:
                continue
            expected = rate * horizon_hours
            n_events = int(np.floor(expected))
            if n_events == 0 and expected >= 0.5:
                n_events = 1
            n_events = min(n_events, self.max_events_per_acuity)
            if n_events == 0:
                continue

            spacing = horizon_hours / (n_events + 1)
            los_hours = self.los_hours_by_acuity.get(
                acuity,
                self.default_los_hours,
            )
            for k in range(1, n_events + 1):
                predictions.append(
                    ArrivalEvent(
                        subject_id=next_subject_id,
                        arrival_time=now + pd.Timedelta(hours=spacing * k),
                        los_hours=los_hours,
                        acuity=acuity,
                    )
                )
                next_subject_id -= 1

        predictions.sort(key=lambda event: event.arrival_time)
        return predictions
