"""Ingest raw MIMIC-IV CSV files into a SQLite cache.

Why this exists
---------------
PhysioNet ships MIMIC-IV (and its Clinical Database Demo) as ``.csv.gz`` files
organised into ``hosp/`` and ``icu/`` sub-directories. Our simulator wants a
SQLite database. Rather than make every user run a conversion script, this
module auto-discovers the relevant CSVs in whatever folder the user unzipped
the dataset into and writes a SQLite cache next to it.

Behaviour
---------
- Works with Demo v2.2 and the full MIMIC-IV (v2.2 / v3.1) releases. They
  share the same ``admissions`` / ``icustays`` schema; only the row counts
  differ.
- Accepts either a flat directory (``admissions.csv.gz`` + ``icustays.csv.gz``
  side-by-side) OR the official layout (``hosp/admissions.csv.gz`` +
  ``icu/icustays.csv.gz``).
- Accepts ``.csv`` OR ``.csv.gz``.
- Reads in chunks, so even full MIMIC-IV (~500k admissions) fits in memory.
- Caches the resulting SQLite at ``<mimic-dir>/.icu_scheduler_cache.sqlite``
  with a fingerprint of the input files. If the fingerprint still matches on
  the next run, we skip re-ingesting. Use ``force=True`` to override.

This is deliberately scoped to the two tables the simulator needs. Extending
to more tables (``patients``, ``labevents``, ...) is one pattern-match away,
but isn't needed for the current course project.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ---------- Public constants --------------------------------------------------

REQUIRED_TABLES: Tuple[str, ...] = ("admissions", "icustays")

#: Columns the downstream loader expects. These are the MIMIC-IV v2.2 / v3.1
#: canonical column names. If a future MIMIC-IV release renames columns, adapt
#: ``_TABLE_SPECS`` below.
_TABLE_SPECS: Dict[str, Dict[str, object]] = {
    "admissions": {
        "columns": [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "hospital_expire_flag",
        ],
        "parse_dates": ["admittime", "dischtime"],
        "dtypes": {
            "subject_id": "Int64",
            "hadm_id": "Int64",
            "hospital_expire_flag": "Int64",
        },
    },
    "icustays": {
        "columns": [
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ],
        "parse_dates": ["intime", "outtime"],
        "dtypes": {
            "subject_id": "Int64",
            "hadm_id": "Int64",
            "stay_id": "Int64",
            "los": "float64",
        },
    },
}

#: Sub-directories we look inside. Covers both the flat layout and the
#: official "modules" layout (``hosp/`` for hospital tables, ``icu/`` for ICU).
_SEARCH_SUBDIRS: Tuple[str, ...] = ("", "hosp", "icu")

CACHE_FILENAME = ".icu_scheduler_cache.sqlite"
FINGERPRINT_FILENAME = ".icu_scheduler_cache.fingerprint.json"

_CHUNKSIZE = 50_000


# ---------- Discovery ---------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredFiles:
    """The CSV paths we found for each required table."""

    admissions: Path
    icustays: Path

    def as_dict(self) -> Dict[str, Path]:
        return {"admissions": self.admissions, "icustays": self.icustays}


def discover_mimic_files(mimic_dir: Path) -> DiscoveredFiles:
    """Find ``admissions`` and ``icustays`` CSVs under ``mimic_dir``.

    Parameters
    ----------
    mimic_dir
        The folder the user unzipped MIMIC-IV into. May contain the files
        directly, or under ``hosp/`` / ``icu/`` subdirectories.

    Returns
    -------
    DiscoveredFiles

    Raises
    ------
    FileNotFoundError
        If ``mimic_dir`` does not exist, or one of the required tables cannot
        be found as either ``<table>.csv`` or ``<table>.csv.gz``.
    """
    mimic_dir = Path(mimic_dir)
    if not mimic_dir.is_dir():
        raise FileNotFoundError(f"MIMIC directory not found: {mimic_dir}")

    found: Dict[str, Path] = {}
    for table in REQUIRED_TABLES:
        path = _find_table_file(mimic_dir, table)
        if path is None:
            searched = ", ".join(
                str(mimic_dir / sub) if sub else str(mimic_dir)
                for sub in _SEARCH_SUBDIRS
            )
            raise FileNotFoundError(
                f"Could not find {table}.csv or {table}.csv.gz under "
                f"{mimic_dir}. Searched: {searched}"
            )
        found[table] = path

    return DiscoveredFiles(
        admissions=found["admissions"],
        icustays=found["icustays"],
    )


def _find_table_file(mimic_dir: Path, table: str) -> Optional[Path]:
    """Return the first matching ``<table>.csv[.gz]`` path, or ``None``.

    Matching is case-insensitive — MIMIC-III's demo ships UPPERCASE CSV names
    (``ADMISSIONS.csv``) while MIMIC-IV uses lowercase; both should work.
    """
    wanted = {
        f"{table}.csv".lower(),
        f"{table}.csv.gz".lower(),
    }
    for subdir in _SEARCH_SUBDIRS:
        base = mimic_dir / subdir if subdir else mimic_dir
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if entry.is_file() and entry.name.lower() in wanted:
                return entry
    return None


# ---------- Fingerprinting ----------------------------------------------------


def _fingerprint(files: DiscoveredFiles) -> Dict[str, Dict[str, object]]:
    """Return a (cheap) fingerprint of the input files for cache invalidation.

    Uses each file's absolute path, size, and mtime. We deliberately do NOT
    hash the contents — full MIMIC-IV is multi-gigabyte, and the mtime+size
    pair is plenty for detecting user swapping one release for another.
    """
    fp: Dict[str, Dict[str, object]] = {}
    for table, path in files.as_dict().items():
        stat = path.stat()
        fp[table] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
    return fp


def _load_fingerprint(fp_path: Path) -> Optional[Dict[str, Dict[str, object]]]:
    if not fp_path.is_file():
        return None
    try:
        return json.loads(fp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_fingerprint(fp_path: Path, fp: Dict[str, Dict[str, object]]) -> None:
    fp_path.write_text(json.dumps(fp, indent=2, sort_keys=True))


# ---------- CSV → SQLite ------------------------------------------------------


def _open_csv(path: Path) -> Iterator[pd.DataFrame]:
    """Yield DataFrames in chunks from a ``.csv`` or ``.csv.gz`` file."""
    # pandas handles gzip transparently based on the suffix.
    return pd.read_csv(path, chunksize=_CHUNKSIZE)


def _ingest_table(
    engine: Engine,
    table: str,
    path: Path,
) -> int:
    """Read one CSV and write it to SQLite in chunks. Returns row count."""
    spec = _TABLE_SPECS[table]
    expected_cols = spec["columns"]
    parse_dates = spec["parse_dates"]
    dtypes = spec["dtypes"]

    total_rows = 0
    first_chunk = True

    reader = pd.read_csv(
        path,
        chunksize=_CHUNKSIZE,
        parse_dates=parse_dates,
        low_memory=False,
    )
    for chunk in reader:
        missing = [c for c in expected_cols if c not in chunk.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing expected columns for '{table}': {missing}. "
                f"Found columns: {list(chunk.columns)}"
            )
        chunk = chunk[expected_cols].copy()
        # Coerce numeric dtypes where specified; tolerate NaN for Int64.
        for col, dt in dtypes.items():
            if dt == "Int64":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
            elif dt == "float64":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk.to_sql(
            table,
            con=engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )
        first_chunk = False
        total_rows += len(chunk)

    # If the CSV was empty (no chunks), still create an empty table with the
    # right schema so downstream queries don't blow up.
    if first_chunk:
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in expected_cols})
        empty.to_sql(table, con=engine, if_exists="replace", index=False)

    return total_rows


# ---------- Public API --------------------------------------------------------


def ingest_mimic_dir(
    mimic_dir: str | Path,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Ingest a MIMIC-IV directory into SQLite, with fingerprint-based caching.

    Parameters
    ----------
    mimic_dir
        Folder the user unzipped MIMIC-IV (or the Demo) into.
    cache_path
        Where to write the SQLite cache. Defaults to
        ``<mimic_dir>/.icu_scheduler_cache.sqlite``.
    force
        If ``True``, rebuild the cache even if the fingerprint matches.

    Returns
    -------
    Path
        Path to the SQLite cache file, ready to feed to
        :func:`icu_scheduler.data_loader.load_joined_cohort`.

    Examples
    --------
    >>> sqlite = ingest_mimic_dir("~/Downloads/mimic-iv-demo")  # doctest: +SKIP
    >>> from icu_scheduler.data_loader import load_joined_cohort
    >>> cohort = load_joined_cohort(sqlite)                     # doctest: +SKIP
    """
    mimic_dir = Path(mimic_dir).expanduser().resolve()
    files = discover_mimic_files(mimic_dir)

    if cache_path is None:
        cache_path = mimic_dir / CACHE_FILENAME
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path = cache_path.with_name(FINGERPRINT_FILENAME)

    fp_new = _fingerprint(files)

    if not force and cache_path.is_file():
        fp_old = _load_fingerprint(fp_path)
        if fp_old == fp_new:
            return cache_path  # cache hit

    # (Re)build the cache. Write to a temp path first so a failed ingest
    # doesn't leave a broken cache behind.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    engine = create_engine(f"sqlite:///{tmp_path}", future=True)
    try:
        for table, path in files.as_dict().items():
            _ingest_table(engine, table, path)
        # Indexes that pay for themselves on the joined query.
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_adm_hadm ON admissions(hadm_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_icu_hadm ON icustays(hadm_id)"))
            conn.commit()
    finally:
        engine.dispose()

    if cache_path.exists():
        cache_path.unlink()
    tmp_path.rename(cache_path)
    _save_fingerprint(fp_path, fp_new)
    return cache_path


def cache_is_fresh(mimic_dir: str | Path, cache_path: Optional[Path] = None) -> bool:
    """Return True if a fingerprint-matching cache already exists."""
    mimic_dir = Path(mimic_dir).expanduser().resolve()
    try:
        files = discover_mimic_files(mimic_dir)
    except FileNotFoundError:
        return False

    if cache_path is None:
        cache_path = mimic_dir / CACHE_FILENAME
    cache_path = Path(cache_path)
    fp_path = cache_path.with_name(FINGERPRINT_FILENAME)

    if not cache_path.is_file():
        return False
    fp_old = _load_fingerprint(fp_path)
    return fp_old == _fingerprint(files)
