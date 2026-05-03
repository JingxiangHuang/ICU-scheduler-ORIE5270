"""Tests for :mod:`icu_scheduler.mimic_ingest`.

We build tiny MIMIC-IV-shaped CSVs in a temp dir and verify:
- flat layout vs. hosp/ + icu/ subdirectory layout both work
- .csv and .csv.gz are both recognised
- a repeat call uses the cache; ``force=True`` / a changed fingerprint rebuilds
- missing columns raise a clear error
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from icu_scheduler.mimic_ingest import (
    CACHE_FILENAME,
    FINGERPRINT_FILENAME,
    cache_is_fresh,
    discover_mimic_files,
    ingest_mimic_dir,
)

# ---------- fixtures ----------------------------------------------------------


ADMISSIONS_COLS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "admission_type",
    "admission_location",
    "discharge_location",
    "hospital_expire_flag",
]
ICUSTAYS_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "first_careunit",
    "last_careunit",
    "intime",
    "outtime",
    "los",
]


def _fake_admissions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "admittime": "2180-01-01 08:00:00",
                "dischtime": "2180-01-05 10:00:00",
                "admission_type": "EW EMER.",
                "admission_location": "EMERGENCY ROOM",
                "discharge_location": "HOME",
                "hospital_expire_flag": 0,
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "admittime": "2180-01-02 12:00:00",
                "dischtime": "2180-01-04 09:00:00",
                "admission_type": "ELECTIVE",
                "admission_location": "CLINIC",
                "discharge_location": "HOME",
                "hospital_expire_flag": 0,
            },
        ]
    )


def _fake_icustays() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "stay_id": 100,
                "first_careunit": "MICU",
                "last_careunit": "MICU",
                "intime": "2180-01-01 09:00:00",
                "outtime": "2180-01-03 09:00:00",
                "los": 2.0,
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "stay_id": 200,
                "first_careunit": "SICU",
                "last_careunit": "SICU",
                "intime": "2180-01-02 14:00:00",
                "outtime": "2180-01-03 18:00:00",
                "los": 1.1667,
            },
        ]
    )


def _write_csv(df: pd.DataFrame, path: Path, gzipped: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


# ---------- discovery ---------------------------------------------------------


def test_discover_flat_layout_csv(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    found = discover_mimic_files(tmp_path)
    assert found.admissions.name == "admissions.csv"
    assert found.icustays.name == "icustays.csv"


def test_discover_flat_layout_gz(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv.gz", gzipped=True)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv.gz", gzipped=True)

    found = discover_mimic_files(tmp_path)
    assert found.admissions.suffixes[-2:] == [".csv", ".gz"]
    assert found.icustays.suffixes[-2:] == [".csv", ".gz"]


def test_discover_official_subdir_layout(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "hosp" / "admissions.csv.gz", gzipped=True)
    _write_csv(_fake_icustays(), tmp_path / "icu" / "icustays.csv.gz", gzipped=True)

    found = discover_mimic_files(tmp_path)
    assert "hosp" in str(found.admissions)
    assert "icu" in str(found.icustays)


def test_discover_missing_file_raises(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    # icustays deliberately missing
    with pytest.raises(FileNotFoundError, match="icustays"):
        discover_mimic_files(tmp_path)


def test_discover_nonexistent_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_mimic_files(tmp_path / "does_not_exist")


# ---------- ingest ------------------------------------------------------------


def test_ingest_flat_layout_writes_sqlite(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    cache = ingest_mimic_dir(tmp_path)
    assert cache.exists()
    assert cache.name == CACHE_FILENAME

    engine = create_engine(f"sqlite:///{cache}", future=True)
    insp = inspect(engine)
    assert {"admissions", "icustays"}.issubset(set(insp.get_table_names()))

    with engine.connect() as conn:
        n_adm = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar_one()
        n_icu = conn.execute(text("SELECT COUNT(*) FROM icustays")).scalar_one()
    assert n_adm == 2
    assert n_icu == 2


def test_ingest_subdir_layout_gz(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "hosp" / "admissions.csv.gz", gzipped=True)
    _write_csv(_fake_icustays(), tmp_path / "icu" / "icustays.csv.gz", gzipped=True)

    cache = ingest_mimic_dir(tmp_path)
    assert cache.exists()

    # Downstream loader should work on this cache.
    from icu_scheduler.data_loader import load_joined_cohort

    cohort = load_joined_cohort(cache)
    assert len(cohort) == 2
    assert {"admission_type", "first_careunit", "intime", "los"}.issubset(cohort.columns)


def test_ingest_caches_on_repeat_call(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    cache = ingest_mimic_dir(tmp_path)
    mtime1 = cache.stat().st_mtime

    # Ensure the OS clock ticks so any rebuild would produce a different mtime.
    time.sleep(1.1)

    cache2 = ingest_mimic_dir(tmp_path)
    assert cache == cache2
    assert cache2.stat().st_mtime == mtime1, "cache should not have been rebuilt"
    assert cache_is_fresh(tmp_path)


def test_force_reingest_rebuilds(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    cache = ingest_mimic_dir(tmp_path)
    mtime1 = cache.stat().st_mtime
    time.sleep(1.1)

    cache2 = ingest_mimic_dir(tmp_path, force=True)
    assert cache2.stat().st_mtime > mtime1


def test_changed_input_invalidates_cache(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    ingest_mimic_dir(tmp_path)
    assert cache_is_fresh(tmp_path)

    # Rewrite admissions with a different row count.
    bigger = pd.concat([_fake_admissions(), _fake_admissions()], ignore_index=True)
    bigger["hadm_id"] = [10, 20, 30, 40]
    time.sleep(1.1)
    _write_csv(bigger, tmp_path / "admissions.csv", gzipped=False)

    assert not cache_is_fresh(tmp_path)


def test_missing_column_raises(tmp_path: Path) -> None:
    bad = _fake_admissions().drop(columns=["admission_type"])
    _write_csv(bad, tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    with pytest.raises(ValueError, match="admission_type"):
        ingest_mimic_dir(tmp_path)


def test_custom_cache_path(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)

    custom = tmp_path / "out" / "my_cache.sqlite"
    cache = ingest_mimic_dir(tmp_path, cache_path=custom)
    assert cache == custom
    assert cache.exists()
    # Fingerprint file should sit next to it, not in mimic_dir.
    assert (custom.with_name(FINGERPRINT_FILENAME)).exists()


def test_fingerprint_file_is_valid_json(tmp_path: Path) -> None:
    _write_csv(_fake_admissions(), tmp_path / "admissions.csv", gzipped=False)
    _write_csv(_fake_icustays(), tmp_path / "icustays.csv", gzipped=False)
    ingest_mimic_dir(tmp_path)

    fp_path = tmp_path / FINGERPRINT_FILENAME
    fp = json.loads(fp_path.read_text())
    assert set(fp.keys()) == {"admissions", "icustays"}
    for entry in fp.values():
        assert {"path", "size", "mtime"}.issubset(entry.keys())
