"""Tests for icu_scheduler.data_loader — see W4 (TDD).

Note: these tests are intentionally written *before* the implementation.
"""

from pathlib import Path

import pandas as pd
import pytest

from icu_scheduler.data_loader import (
    LoaderConfig,
    load_admissions,
    load_icu_stays,
    load_joined_cohort,
)


class TestLoaderConfig:
    def test_url_from_sqlite_path(self, tmp_path):
        cfg = LoaderConfig(sqlite_path=tmp_path / "x.sqlite")
        assert cfg.to_url().startswith("sqlite:///")

    def test_url_from_url_string(self):
        cfg = LoaderConfig(sqlalchemy_url="postgresql://u:p@h/db")
        assert cfg.to_url() == "postgresql://u:p@h/db"

    def test_raises_when_no_source(self):
        with pytest.raises(ValueError):
            LoaderConfig().to_url()


class TestLoadIcuStays:
    def test_returns_dataframe(self, tiny_mimic_db):
        df = load_icu_stays(tiny_mimic_db)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tiny_mimic_db):
        df = load_icu_stays(tiny_mimic_db)
        required = {"subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"}
        assert required.issubset(df.columns)

    def test_intime_is_datetime(self, tiny_mimic_db):
        df = load_icu_stays(tiny_mimic_db)
        assert pd.api.types.is_datetime64_any_dtype(df["intime"])

    def test_ordered_by_intime(self, tiny_mimic_db):
        df = load_icu_stays(tiny_mimic_db)
        assert df["intime"].is_monotonic_increasing

    def test_los_is_non_negative(self, tiny_mimic_db):
        df = load_icu_stays(tiny_mimic_db)
        assert (df["los"] >= 0).all()

    def test_accepts_path_object(self, tmp_path, tiny_mimic_db):
        # tiny_mimic_db returns a URL; make sure Path input also works
        sqlite_path = Path(tiny_mimic_db.replace("sqlite:///", ""))
        df = load_icu_stays(sqlite_path)
        assert len(df) == 5


class TestLoadAdmissions:
    def test_returns_expected_rows(self, tiny_mimic_db):
        df = load_admissions(tiny_mimic_db)
        assert len(df) == 5

    def test_has_admission_type(self, tiny_mimic_db):
        df = load_admissions(tiny_mimic_db)
        assert "admission_type" in df.columns


class TestJoinedCohort:
    def test_joins_admissions_and_icu(self, tiny_mimic_db):
        df = load_joined_cohort(tiny_mimic_db)
        # Every row should have fields from both tables
        assert {"admission_type", "first_careunit", "los"}.issubset(df.columns)

    def test_min_los_filter(self, tiny_mimic_db):
        df = load_joined_cohort(tiny_mimic_db, min_los_hours=30)
        # 30 hours = 1.25 days → only rows with los >= 1.25 survive
        assert (df["los"] >= 1.25).all()
