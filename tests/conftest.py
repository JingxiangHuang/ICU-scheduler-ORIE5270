"""Shared pytest fixtures.

Keeping DB construction in a fixture means:
- No tests talk to real MIMIC-IV
- Tests are ~milliseconds — great for a fast TDD loop
- Coverage of data_loader.py stays high
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def tiny_mimic_db(tmp_path: Path) -> str:
    """Create a minimal in-memory-ish SQLite DB with 5 fake ICU stays.

    Returns a SQLAlchemy URL string pointing at a file in ``tmp_path``.
    """
    db_file = tmp_path / "mimic.sqlite"
    engine = create_engine(f"sqlite:///{db_file}", future=True)

    admissions = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [101, 102, 103, 104, 105],
            "admittime": pd.to_datetime(
                [
                    "2180-01-01 10:00",
                    "2180-01-02 04:30",
                    "2180-01-03 18:00",
                    "2180-01-04 09:00",
                    "2180-01-05 23:30",
                ]
            ),
            "dischtime": pd.to_datetime(
                [
                    "2180-01-03 10:00",
                    "2180-01-04 04:30",
                    "2180-01-06 18:00",
                    "2180-01-06 09:00",
                    "2180-01-07 23:30",
                ]
            ),
            "admission_type": [
                "EMERGENCY",
                "ELECTIVE",
                "EMERGENCY",
                "URGENT",
                "ELECTIVE",
            ],
            "admission_location": ["EMERGENCY ROOM"] * 5,
            "discharge_location": ["HOME"] * 5,
            "hospital_expire_flag": [0, 0, 0, 1, 0],
        }
    )
    icustays = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [101, 102, 103, 104, 105],
            "stay_id": [1001, 1002, 1003, 1004, 1005],
            "first_careunit": ["MICU", "SICU", "MICU", "CCU", "MICU"],
            "last_careunit": ["MICU", "SICU", "MICU", "CCU", "MICU"],
            "intime": pd.to_datetime(
                [
                    "2180-01-01 11:00",
                    "2180-01-02 05:00",
                    "2180-01-03 19:00",
                    "2180-01-04 10:00",
                    "2180-01-06 00:00",
                ]
            ),
            "outtime": pd.to_datetime(
                [
                    "2180-01-02 23:00",
                    "2180-01-03 12:00",
                    "2180-01-05 19:00",
                    "2180-01-05 10:00",
                    "2180-01-07 00:00",
                ]
            ),
            "los": [1.5, 1.29, 2.0, 1.0, 1.0],
        }
    )

    with engine.begin() as conn:
        admissions.to_sql("admissions", conn, index=False, if_exists="replace")
        icustays.to_sql("icustays", conn, index=False, if_exists="replace")

    return f"sqlite:///{db_file}"
