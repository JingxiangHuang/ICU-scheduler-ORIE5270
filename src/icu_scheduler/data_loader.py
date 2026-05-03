"""SQL-backed data loader for MIMIC-IV.

Exercises course skills: W9-W10 (SQL, joins, window functions) and W3-W4 (testable
pure functions wrapping I/O).

Design notes
------------
- Every public function takes a SQLAlchemy URL string, so tests can point at an
  in-memory SQLite database. This keeps the data layer unit-testable without
  requiring MIMIC-IV credentials.
- Queries are stored as module-level constants for transparency.
- DataFrames returned have stable column names — downstream modules depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------- SQL queries -------------------------------------------------------

ICU_STAYS_QUERY = """
SELECT
    s.subject_id,
    s.hadm_id,
    s.stay_id,
    s.first_careunit,
    s.last_careunit,
    s.intime,
    s.outtime,
    s.los
FROM icustays s
ORDER BY s.intime
"""

ADMISSIONS_QUERY = """
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.hospital_expire_flag
FROM admissions a
ORDER BY a.admittime
"""


@dataclass(frozen=True)
class LoaderConfig:
    """Loader configuration. Keep this immutable for easy testing."""

    sqlite_path: Optional[Path] = None
    sqlalchemy_url: Optional[str] = None

    def to_url(self) -> str:
        """Return a SQLAlchemy connection URL.

        Raises
        ------
        ValueError
            If neither ``sqlite_path`` nor ``sqlalchemy_url`` is provided.
        """
        if self.sqlalchemy_url:
            return self.sqlalchemy_url
        if self.sqlite_path is not None:
            return f"sqlite:///{self.sqlite_path}"
        raise ValueError("LoaderConfig requires either sqlite_path or sqlalchemy_url")


# ---------- Public API --------------------------------------------------------


def _make_engine(source: str | Path | LoaderConfig) -> Engine:
    """Normalize a string/Path/LoaderConfig into a SQLAlchemy Engine."""
    if isinstance(source, LoaderConfig):
        url = source.to_url()
    elif isinstance(source, Path):
        url = f"sqlite:///{source}"
    elif isinstance(source, str):
        url = source if "://" in source else f"sqlite:///{source}"
    else:
        raise TypeError(f"Unsupported source type: {type(source)!r}")
    return create_engine(url, future=True)


def load_icu_stays(source: str | Path | LoaderConfig) -> pd.DataFrame:
    """Load ICU stays from MIMIC-IV.

    Parameters
    ----------
    source
        SQLite file path, SQLAlchemy URL, or :class:`LoaderConfig`.

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id``, ``hadm_id``, ``stay_id``, ``first_careunit``,
        ``last_careunit``, ``intime``, ``outtime``, ``los``. Datetime columns
        are parsed to ``pandas.Timestamp``.

    Examples
    --------
    >>> df = load_icu_stays("data/sample/mimic_demo.sqlite")  # doctest: +SKIP
    >>> {"intime", "los"}.issubset(df.columns)                # doctest: +SKIP
    True
    """
    engine = _make_engine(source)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(ICU_STAYS_QUERY),
            conn,
            parse_dates=["intime", "outtime"],
        )
    df = df.dropna(subset=["los"]).reset_index(drop=True)
    return df


def load_admissions(source: str | Path | LoaderConfig) -> pd.DataFrame:
    """Load hospital admissions table.

    Parameters
    ----------
    source
        SQLite file path, SQLAlchemy URL, or :class:`LoaderConfig`.

    Returns
    -------
    pd.DataFrame
        Columns from the admissions table with datetime parsing applied.
    """
    engine = _make_engine(source)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(ADMISSIONS_QUERY),
            conn,
            parse_dates=["admittime", "dischtime"],
        )
    return df.reset_index(drop=True)


_JOINED_COHORT_QUERY = """
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.hospital_expire_flag,
    s.stay_id,
    s.first_careunit,
    s.last_careunit,
    s.intime,
    s.outtime,
    s.los
FROM admissions a
JOIN icustays s ON a.hadm_id = s.hadm_id
WHERE s.los * 24.0 >= :min_los_hours
ORDER BY a.admittime
"""


def load_joined_cohort(
    source: str | Path | LoaderConfig,
    min_los_hours: float = 0.0,
) -> pd.DataFrame:
    """Return admissions joined with their first ICU stay.

    Uses a SQL JOIN (not a pandas merge) to exercise course SQL skills.

    Parameters
    ----------
    source
        SQLite file path, SQLAlchemy URL, or :class:`LoaderConfig`.
    min_los_hours
        Minimum length-of-stay in hours. Rows with shorter stays are excluded.

    Returns
    -------
    pd.DataFrame
        Joined cohort with columns from both tables.
    """
    engine = _make_engine(source)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_JOINED_COHORT_QUERY),
            conn,
            params={"min_los_hours": float(min_los_hours)},
            parse_dates=["admittime", "dischtime", "intime", "outtime"],
        )
    return df.reset_index(drop=True)
