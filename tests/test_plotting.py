"""Smoke tests for icu_scheduler.plotting — verify PNGs get created."""

import numpy as np

from icu_scheduler.plotting import plot_occupancy_bands, plot_policy_comparison
from icu_scheduler.simulator import SimulationResult


def _mk_result(n=10, util_mean=0.5):
    return SimulationResult(
        n_admitted=n,
        n_boarded=0,
        n_diverted=0,
        max_occupancy=5,
        utilization_series=np.full(24, util_mean),
    )


class TestPlotOccupancyBands:
    def test_writes_png(self, tmp_path):
        out = tmp_path / "occupancy.png"
        results = [_mk_result() for _ in range(5)]
        returned = plot_occupancy_bands(results, out)
        assert returned == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_empty_results(self, tmp_path):
        out = tmp_path / "empty.png"
        plot_occupancy_bands([], out)
        assert out.exists()


class TestPlotPolicyComparison:
    def test_writes_png(self, tmp_path):
        out = tmp_path / "comparison.png"
        named = {
            "fcfs": [_mk_result(n=10)],
            "threshold": [_mk_result(n=8)],
        }
        plot_policy_comparison(named, out)
        assert out.exists()
        assert out.stat().st_size > 0
