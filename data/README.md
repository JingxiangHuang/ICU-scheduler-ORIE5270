# Data folder

Drop a MIMIC-IV dataset here and `icu-scheduler run-demo` will find it
automatically — no command-line flags required.

## Quickstart

Download MIMIC-IV (or the free Clinical Database Demo v2.2) from
[PhysioNet](https://physionet.org/content/mimic-iv-demo/), unzip into this
folder, then from the project root run:

```bash
icu-scheduler run-demo
```

That's it. The tool scans `data/` and its immediate subfolders for
`admissions.csv[.gz]` and `icustays.csv[.gz]`. Both of these layouts work:

```
# Layout A — put the files directly in data/
data/
├── admissions.csv.gz
└── icustays.csv.gz

# Layout B — official PhysioNet layout inside a dataset subfolder
data/
└── mimic-iv-clinical-database-demo-2.2/
    ├── hosp/
    │   └── admissions.csv.gz
    └── icu/
        └── icustays.csv.gz
```

Layout B is what you get if you just unzip the PhysioNet archive into
`data/`. Nothing to move, nothing to rename.

## Supported datasets

- **MIMIC-IV Clinical Database Demo v2.2** (public, no credentialing) — ✅
- **MIMIC-IV v2.2 / v3.1 full release** (credentialed access) — ✅
  (same schema as the demo, just more rows)

Any other CSV layout / dataset is fine too if you rename the files to
`admissions.csv[.gz]` and `icustays.csv[.gz]` with the MIMIC-IV column names.

## What happens on first run

1. CSVs are read in chunks (50k rows at a time) and ingested into a SQLite
   cache at `<dataset-folder>/.icu_scheduler_cache.sqlite`.
2. A fingerprint file (`.icu_scheduler_cache.fingerprint.json`) records the
   size + mtime of each input file.
3. Next time you run `icu-scheduler run-demo`, the cache is reused as long as
   the fingerprints still match — ingest is O(milliseconds) instead of
   O(seconds).

Force a rebuild with `icu-scheduler run-demo --force-reingest`.

## Pointing at a dataset somewhere else

If your MIMIC folder lives outside this repo:

```bash
icu-scheduler run-demo --mimic-dir /path/to/mimic-iv/
```

## `sample/`

Contains a tiny pre-built `mimic_demo.sqlite` used by the test suite and the
zero-config fallback. Not for real analysis.
