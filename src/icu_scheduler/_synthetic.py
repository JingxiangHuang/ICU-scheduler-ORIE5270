"""Synthetic MIMIC-IV-shaped SQLite DB.

Lets ``icu-scheduler run-demo`` work zero-setup (no PhysioNet account needed).
Real users are expected to point ``--db`` at a real MIMIC-IV-Demo SQLite file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def build_synthetic_db(
    out_path: Path,
    n_patients: int = 200,
    seed: int = 0,
) -> Path:
    """Build a synthetic SQLite DB that mirrors MIMIC-IV's admissions/icustays.

    The arrival process is a Poisson stream over ~60 days with lognormal LOS
    and a realistic emergency/elective mix.
    """
    out_path = Path(out_path)
    rng = np.random.default_rng(seed)

    # Inter-arrival times in hours (Poisson ~ every 8 hours on average).
    inter = rng.exponential(scale=8.0, size=n_patients)
    arrival_hours = np.cumsum(inter)
    admittime = pd.Timestamp("2180-01-01") + pd.to_timedelta(arrival_hours, unit="h")

    # LOS (days) — lognormal, realistic ICU shape
    los_days = rng.lognormal(mean=0.5, sigma=0.8, size=n_patients)
    los_days = np.clip(los_days, 0.2, 30.0)

    admission_type = rng.choice(
        ["EMERGENCY", "URGENT", "ELECTIVE"],
        size=n_patients,
        p=[0.55, 0.15, 0.30],
    )
    careunit = rng.choice(
        ["MICU", "SICU", "CCU", "NICU", "TSICU"],
        size=n_patients,
        p=[0.45, 0.25, 0.15, 0.05, 0.10],
    )

    subject_id = np.arange(1, n_patients + 1)
    hadm_id = 100000 + subject_id
    stay_id = 200000 + subject_id

    admissions = pd.DataFrame(
        {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "admittime": admittime,
            "dischtime": admittime + pd.to_timedelta(los_days + 0.5, unit="D"),
            "admission_type": admission_type,
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME",
            "hospital_expire_flag": rng.binomial(1, 0.08, size=n_patients),
        }
    )

    icustays = pd.DataFrame(
        {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            "first_careunit": careunit,
            "last_careunit": careunit,
            "intime": admittime + pd.to_timedelta(1.0, unit="h"),
            "outtime": admittime + pd.to_timedelta(1.0, unit="h")
            + pd.to_timedelta(los_days, unit="D"),
            "los": los_days,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{out_path}", future=True)
    with engine.begin() as conn:
        admissions.to_sql("admissions", conn, index=False, if_exists="replace")
        icustays.to_sql("icustays", conn, index=False, if_exists="replace")
    return out_path
