"""Tests for deployable arrival forecasting."""

import pandas as pd
import pytest

from icu_scheduler.forecasting import HistoricalArrivalForecaster
from icu_scheduler.stream import ArrivalEvent


def _event(subject_id, hour, los_hours, acuity):
    return ArrivalEvent(
        subject_id=subject_id,
        arrival_time=pd.Timestamp("2180-01-01") + pd.Timedelta(hours=hour),
        los_hours=los_hours,
        acuity=acuity,
    )


class TestHistoricalArrivalForecaster:
    def test_fit_estimates_rates_and_los_by_acuity(self):
        forecaster = HistoricalArrivalForecaster.from_events(
            [
                _event(1, 0, 24, "urgent"),
                _event(2, 10, 48, "urgent"),
                _event(3, 20, 12, "elective"),
            ]
        )

        assert forecaster.rates_per_hour["urgent"] == pytest.approx(2 / 20)
        assert forecaster.los_hours_by_acuity["urgent"] == pytest.approx(36)

    def test_predict_generates_future_events_without_true_future_stream(self):
        forecaster = HistoricalArrivalForecaster(
            rates_per_hour={"urgent": 1 / 12},
            los_hours_by_acuity={"urgent": 24},
            default_los_hours=24,
        )

        predictions = forecaster.predict(pd.Timestamp("2180-01-01"), horizon_hours=24)

        assert len(predictions) == 2
        assert all(event.acuity == "urgent" for event in predictions)
        assert all(event.los_hours == 24 for event in predictions)

    def test_predict_rejects_non_positive_horizon(self):
        forecaster = HistoricalArrivalForecaster({}, {}, default_los_hours=24)

        with pytest.raises(ValueError):
            forecaster.predict(pd.Timestamp("2180-01-01"), horizon_hours=0)
