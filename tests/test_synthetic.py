"""Tests for icu_scheduler._synthetic — verify synthetic DB is well-formed."""

import pandas as pd
from sqlalchemy import create_engine, inspect

from icu_scheduler._synthetic import build_synthetic_db
from icu_scheduler.data_loader import load_icu_stays, load_joined_cohort


class TestSyntheticDb:
    def test_builds_expected_tables(self, tmp_path):
        out = tmp_path / "synth.sqlite"
        build_synthetic_db(out, n_patients=20, seed=123)
        assert out.exists()

        engine = create_engine(f"sqlite:///{out}", future=True)
        insp = inspect(engine)
        assert {"admissions", "icustays"}.issubset(set(insp.get_table_names()))

    def test_integrates_with_data_loader(self, tmp_path):
        out = tmp_path / "synth.sqlite"
        build_synthetic_db(out, n_patients=30, seed=7)
        stays = load_icu_stays(out)
        assert len(stays) == 30

    def test_joined_cohort_non_empty(self, tmp_path):
        out = tmp_path / "synth.sqlite"
        build_synthetic_db(out, n_patients=25, seed=1)
        cohort = load_joined_cohort(out)
        assert len(cohort) == 25
        assert "admission_type" in cohort.columns
        assert "first_careunit" in cohort.columns

    def test_determinism(self, tmp_path):
        a = tmp_path / "a.sqlite"
        b = tmp_path / "b.sqlite"
        build_synthetic_db(a, n_patients=15, seed=42)
        build_synthetic_db(b, n_patients=15, seed=42)

        df_a = load_icu_stays(a)
        df_b = load_icu_stays(b)
        pd.testing.assert_frame_equal(df_a, df_b)
