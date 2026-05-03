"""Evaluate scheduling policies on the real MIMIC-IV-Demo v2.2 dataset.

Why this script exists (separately from ``icu-scheduler run-demo``)
------------------------------------------------------------------

``icu-scheduler run-demo`` is a general-purpose entry point. This script is
the **authoritative real-data benchmark** for the project write-up:

* Tuned parameters (``n_beds=6``, 4-way reserve sweep) that put the policies
  into the operating regime where their differences are actually visible on
  the MIMIC-IV-Demo cohort.
* Timeline compression (MIMIC-IV deidentifies by shifting each patient's
  timestamps by ~100 years; without compression the 140 arrivals would look
  91 years apart and no bed contention ever happens).
* More Monte-Carlo replications (200 vs. the CLI default 100).
* Extra diagnostics (acuity distribution, careunit bed-days) printed to
  stdout so the script doubles as a reproducibility check.

Typical usage
-------------
From the project root, after ``pip install -e .`` and placing the
MIMIC-IV-Demo folder under ``data/``::

    python scripts/run_real_mimic.py

Outputs land in ``reports/real_mimic/``.

This script intentionally mirrors what ``icu-scheduler run-demo
--compress-timeline --n-beds 6 --reserves 1,2,3`` does, plus a few extra
diagnostics. The CLI command is the short form; this script is the written
record.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from icu_scheduler.data_loader import load_joined_cohort
from icu_scheduler.evaluator import compare_policies
from icu_scheduler.mapreduce import aggregate_careunit_hours
from icu_scheduler.mimic_ingest import discover_mimic_files, ingest_mimic_dir
from icu_scheduler.plotting import plot_occupancy_bands, plot_policy_comparison
from icu_scheduler.scheduler import FCFSPolicy, ThresholdPolicy
from icu_scheduler.simulator import MonteCarloSimulator
from icu_scheduler.stream import ArrivalStream, _normalize_acuity


# ---------- configuration -----------------------------------------------------

#: Where to look for the MIMIC-IV-Demo folder. We try the conventional
#: ``data/mimic-iv-clinical-database-demo-2.2/`` first and fall back to a flat
#: ``data/`` layout. Pass ``--mimic-dir`` on the command line to override.
DEFAULT_CANDIDATES = [
    Path("data/mimic-iv-clinical-database-demo-2.2"),
    Path("data/mimic-iv-demo"),
    Path("data"),
]

#: Tuned so FCFS admits ~99% while Threshold r=3 has to divert a few percent.
N_BEDS = 5
RESERVES = (1, 2, 3, 4)
MAX_BOARD = 2
N_RUNS = 200
N_WORKERS = 4
OUTPUT_DIR = Path("reports/real_mimic")


# ---------- helpers -----------------------------------------------------------


def _find_mimic_dir() -> Path:
    """Return the first candidate that contains a MIMIC-IV dataset."""
    for candidate in DEFAULT_CANDIDATES:
        if not candidate.is_dir():
            continue
        try:
            discover_mimic_files(candidate)
        except FileNotFoundError:
            continue
        return candidate
    raise FileNotFoundError(
        "No MIMIC-IV dataset found. Unzip MIMIC-IV-Demo v2.2 into "
        "data/mimic-iv-clinical-database-demo-2.2/ and re-run."
    )


def _compress_timeline(cohort: pd.DataFrame) -> pd.DataFrame:
    """Re-anchor all arrivals to a unified timeline.

    MIMIC-IV shifts each patient by a random ~100-year offset, so naive
    playback would spread 140 arrivals across 91 calendar years and ensure no
    bed ever fills up. We preserve inter-arrival gaps (capped at 72h to
    emulate a busier ward) and re-anchor everyone to 2180-01-01.
    """
    cohort = cohort.sort_values("intime").reset_index(drop=True)
    deltas = (cohort["intime"] - cohort["intime"].shift(1)).dt.total_seconds() / 3600
    deltas = deltas.fillna(0).clip(upper=24)
    unified_start = pd.Timestamp("2180-01-01")
    cohort = cohort.copy()
    cohort["intime"] = unified_start + pd.to_timedelta(deltas.cumsum(), unit="h")
    return cohort


# ---------- main --------------------------------------------------------------


def main() -> None:
    mimic_dir = _find_mimic_dir()

    print("=" * 72)
    print("ICU SCHEDULER — Real MIMIC-IV-Demo v2.2 evaluation")
    print("=" * 72)
    print(f"[0] MIMIC directory: {mimic_dir}")

    # 1. Ingest (cache-aware: re-runs are instant)
    db = ingest_mimic_dir(mimic_dir)
    print(f"[1] SQLite cache:   {db}")

    # 2. Load joined cohort
    cohort = load_joined_cohort(db)
    print(f"[2] Cohort loaded:  {len(cohort)} ICU stays, "
          f"{cohort['subject_id'].nunique()} unique patients")

    # 3. Timeline compression
    cohort = _compress_timeline(cohort)
    span = (cohort["intime"].max() - cohort["intime"].min()).days
    print(f"[3] Timeline compressed → spans {span} days "
          f"({cohort['intime'].min()} → {cohort['intime'].max()})")

    # 4. LOS summary
    print(f"[4] LOS (days): mean={cohort['los'].mean():.2f}, "
          f"median={cohort['los'].median():.2f}, "
          f"p95={cohort['los'].quantile(0.95):.2f}")

    # 5. Acuity distribution (after normalization)
    acuities = Counter(_normalize_acuity(t) for t in cohort["admission_type"])
    print("[5] Acuity distribution after normalization:")
    for k, v in sorted(acuities.items(), key=lambda x: -x[1]):
        print(f"      {k:<15} {v:>4} ({100 * v / len(cohort):.1f}%)")

    # 6. MapReduce: total ICU-hours by careunit
    cohort_for_mr = cohort.assign(
        first_careunit=cohort["first_careunit"].str.split(" ").str[0]
    )
    careunit_hours = aggregate_careunit_hours(cohort_for_mr, n_workers=1)
    print("[6] Total ICU-hours by careunit (MapReduce):")
    for unit, hours in sorted(careunit_hours.items(), key=lambda kv: -kv[1]):
        print(f"      {unit:<15} {hours:>8.1f} h ({hours / 24:.1f} bed-days)")

    # 7. Build stream
    cohort = cohort.assign(los_hours=cohort["los"] * 24.0)
    stream = ArrivalStream.from_dataframe(cohort)
    print(f"[7] ArrivalStream built with {len(stream)} events")

    # 8. Policy grid
    policies = {"FCFS": FCFSPolicy(max_board=MAX_BOARD, n_beds=N_BEDS)}
    for r in RESERVES:
        policies[f"Threshold_r{r}"] = ThresholdPolicy(
            n_beds=N_BEDS, reserve=r, max_board=MAX_BOARD
        )

    print(f"\n[8] Running Monte-Carlo: {N_RUNS} runs × {len(policies)} policies "
          f"({N_WORKERS} workers, n_beds={N_BEDS})...")

    named_results = {}
    for name, policy in policies.items():
        sim = MonteCarloSimulator(
            policy=policy, stream=stream,
            n_runs=N_RUNS, n_workers=N_WORKERS,
            perturb=True, seed=42,
        )
        named_results[name] = sim.run()

    # 9. KPIs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    metrics = compare_policies(named_results)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(f"\n[9] Results — written to {metrics_path}")
    print("-" * 72)
    print(f"{'Policy':<15} {'MeanUtil':>9} {'P95Util':>9} "
          f"{'Admit%':>8} {'Board%':>8} {'Divert%':>8} {'MaxOcc':>7} "
          f"{'UrgAdmt%':>9} {'EleAdmt%':>9}")
    print("-" * 92)
    for _, row in metrics.iterrows():
        print(f"{row['policy']:<15} "
              f"{row['mean_utilization'] * 100:>8.1f}% "
              f"{row['p95_utilization'] * 100:>8.1f}% "
              f"{row['admit_rate'] * 100:>7.1f}% "
              f"{row['board_rate'] * 100:>7.1f}% "
              f"{row['divert_rate'] * 100:>7.1f}% "
              f"{row['max_occupancy']:>7.1f} "
              f"{row['urgent_admit_rate'] * 100:>8.1f}% "
              f"{row['elective_admit_rate'] * 100:>8.1f}%")

    # 10. Figures
    plot_policy_comparison(named_results, OUTPUT_DIR / "figures/comparison.png")
    for name, results in named_results.items():
        plot_occupancy_bands(results, OUTPUT_DIR / f"figures/occupancy_{name}.png")

    print(f"\n[10] Figures written to {OUTPUT_DIR / 'figures'}/")
    print("     - comparison.png")
    for name in named_results:
        print(f"     - occupancy_{name}.png")

    print("\n" + "=" * 72)
    print("DONE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
