"""Smoke tests for ``scripts/run_real_mimic.py``.

We don't actually run it against the full 200 × 4 grid — that would take
minutes. We just make sure the module imports cleanly and its helpers work,
which catches the common regressions (broken imports, renamed APIs, etc.).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_real_mimic.py"
STRESS_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "stress_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_real_mimic", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_real_mimic"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_stress_module():
    spec = importlib.util.spec_from_file_location("stress_sweep", STRESS_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stress_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_file_exists():
    assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} missing"


def test_script_imports_cleanly():
    mod = _load_module()
    # Every public constant used by the docstring should exist.
    assert hasattr(mod, "N_BEDS")
    assert hasattr(mod, "RESERVES")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_compress_timeline_monotonic_and_capped():
    """The timeline-compression helper is the scientifically-important bit."""
    mod = _load_module()
    # Two arrivals 100 years apart; after compression the gap should be
    # clipped to at most 72h.
    df = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "intime": [pd.Timestamp("2100-01-01"), pd.Timestamp("2200-01-01")],
            "los": [1.0, 1.0],
            "first_careunit": ["MICU", "MICU"],
            "admission_type": ["EW EMER.", "ELECTIVE"],
        }
    )
    out = mod._compress_timeline(df)
    gap_hours = (out["intime"].iloc[1] - out["intime"].iloc[0]).total_seconds() / 3600
    assert gap_hours <= 72.1
    assert out["intime"].is_monotonic_increasing


def test_find_mimic_dir_raises_when_no_data(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    # Replace the candidate list with something guaranteed empty.
    monkeypatch.setattr(mod, "DEFAULT_CANDIDATES", [tmp_path / "nonexistent"])
    with pytest.raises(FileNotFoundError):
        mod._find_mimic_dir()


# ---------- stress_sweep.py --------------------------------------------------


def test_stress_script_imports_cleanly():
    """The stress sweep script's helpers must stay importable."""
    mod = _load_stress_module()
    assert hasattr(mod, "BEDS_GRID")
    assert hasattr(mod, "RATES_GRID")
    assert hasattr(mod, "RESERVES")
    assert callable(mod.main)
    assert callable(mod._prepare_cohort)


def test_stress_grid_axes_are_sensible():
    """Smoke check on grid choices — we want enough points to draw a heatmap."""
    mod = _load_stress_module()
    assert len(mod.BEDS_GRID) >= 3
    assert len(mod.RATES_GRID) >= 3
    # rate=1.0 must be present so users can compare against historical pace.
    assert 1.0 in mod.RATES_GRID
    # beds must include a low value where contention actually happens.
    assert min(mod.BEDS_GRID) <= 4


def test_prepare_cohort_scales_density(tiny_mimic_db):
    """Doubling the rate should approximately halve the timeline span."""
    mod = _load_stress_module()
    db_path = tiny_mimic_db.replace("sqlite:///", "")

    cohort_1x = mod._prepare_cohort(db_path, rate_multiplier=1.0)
    cohort_2x = mod._prepare_cohort(db_path, rate_multiplier=2.0)

    span_1x = (cohort_1x["intime"].max() - cohort_1x["intime"].min()).total_seconds()
    span_2x = (cohort_2x["intime"].max() - cohort_2x["intime"].min()).total_seconds()
    # Same number of arrivals, same anchor, half the gaps -> ~half the span.
    # Allow generous tolerance because the 5-row fixture is tiny.
    assert span_2x == pytest.approx(span_1x / 2.0, rel=0.05)
    assert len(cohort_2x) == len(cohort_1x)


def test_prepare_cohort_preserves_ordering(tiny_mimic_db):
    """After scaling, intimes must still be monotonically increasing."""
    mod = _load_stress_module()
    db_path = tiny_mimic_db.replace("sqlite:///", "")
    cohort = mod._prepare_cohort(db_path, rate_multiplier=3.0)
    assert cohort["intime"].is_monotonic_increasing
    assert "los_hours" in cohort.columns
    # LOS itself is unchanged — this is a pure arrival-rate stress test.
    assert (cohort["los_hours"] > 0).all()
