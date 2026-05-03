"""Build the bundled SQLite demo database from MIMIC-IV-Demo CSV files.

Usage
-----
First download MIMIC-IV-Demo v2.2 from PhysioNet:
    https://physionet.org/content/mimic-iv-demo/2.2/

Unzip into ``data/raw/``, then:

    python scripts/build_sample_db.py \
        --raw data/raw/mimic-iv-demo-2.2 \
        --out data/sample/mimic_demo.sqlite

The resulting SQLite file is ~5 MB and is committed to the repo so any grader
can run the demo without PhysioNet credentials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


TABLES_TO_INGEST = {
    "admissions": "hosp/admissions.csv.gz",
    "patients": "hosp/patients.csv.gz",
    "icustays": "icu/icustays.csv.gz",
    "transfers": "hosp/transfers.csv.gz",
}


def build(raw_dir: Path, out_path: Path) -> None:
    """Read CSV.gz files from ``raw_dir`` and write them to a SQLite file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{out_path}", future=True)

    for table, rel_path in TABLES_TO_INGEST.items():
        csv_path = raw_dir / rel_path
        print(f"[build_sample_db] Loading {csv_path.name} → table '{table}'")
        df = pd.read_csv(csv_path, compression="gzip")
        with engine.begin() as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[build_sample_db] Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="MIMIC-IV-Demo root folder")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/sample/mimic_demo.sqlite"),
        help="Output SQLite file",
    )
    args = ap.parse_args()
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
