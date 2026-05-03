"""Stress-sweep: where do scheduling policies actually differ?

Why this script exists
----------------------

Running the policies at historical pace on the MIMIC-IV-Demo cohort gives
admission rates clustered around 95-99% — the differences are real but
visually unimpressive. That's not a bug in the algorithms; it's that the
*operating regime* matters. Queueing theory tells us policy gaps explode as
load factor ρ = λ/μ approaches 1. Below 0.5 they're invisible; above 0.9
they're dramatic.

This script sweeps the two parameters that control ρ — bed count ``n_beds``
and arrival-rate multiplier — and produces:

* ``stress_grid.csv``: full KPI table for every (beds, rate, policy) cell.
* ``figures/divert_gap.png``: heatmap of (FCFS divert rate - best reservation
  policy divert rate). Bright cells = where reservation policy genuinely wins.
* ``figures/urgent_protection.png``: heatmap of urgent-admit-rate gap. The
  point of reservation policies is to protect urgent patients by holding back
  beds; this figure makes that protection visible.
* ``figures/admit_curves.png``: admit-rate vs arrival-rate, one line per
  policy, faceted by bed count. The "scissor" at high rates is the headline.

Typical usage
-------------
From the project root::

    python scripts/stress_sweep.py

Outputs land in ``reports/stress_sweep/``. Runtime is a short smoke run by
default; increase ``N_RUNS`` for final reported numbers.

Tuning knobs are at the top of the file (``BEDS_GRID``, ``RATES_GRID``,
``RESERVES``, ``N_RUNS``) — turn them down for a faster smoke run, turn
``N_RUNS`` up for tighter confidence intervals in the write-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from icu_scheduler.data_loader import load_joined_cohort
from icu_scheduler.evaluator import compute_kpis
from icu_scheduler.mimic_ingest import ingest_mimic_dir
from icu_scheduler.scheduler import (
    AdaptiveThresholdPolicy,
    FCFSPolicy,
    ThresholdPolicy,
)
from icu_scheduler.simulator import MonteCarloSimulator
from icu_scheduler.stream import ArrivalStream


# ---------- configuration -----------------------------------------------------

#: Bed-count axis. The Demo cohort has ~140 stays; below 4 beds even the best
#: policy diverts heavily, above 10 beds nothing fills up. The interesting
#: regime is 3-8.
BEDS_GRID: tuple = (3, 4, 5, 6, 8)

#: Arrival-rate multiplier axis. 1.0 = historical pace. 4.0 = brutal surge.
RATES_GRID: tuple = (1.0, 1.5, 2.0, 3.0, 4.0)

#: Fixed ThresholdPolicy reservation values to compare against FCFS/adaptive.
RESERVES: tuple = (1, 2, 3)

MAX_BOARD = 3
N_RUNS = 10
N_WORKERS = 4
OUTPUT_DIR = Path("reports/stress_sweep")

DEFAULT_CANDIDATES = [
    Path("data/mimic-iv-clinical-database-demo-2.2"),
    Path("data/mimic-iv-demo"),
    Path("data"),
]


# ---------- helpers -----------------------------------------------------------


def _find_mimic_dir() -> Path:
    """Return the first candidate that has a MIMIC-IV-shaped dataset."""
    from icu_scheduler.mimic_ingest import discover_mimic_files

    for cand in DEFAULT_CANDIDATES:
        if not cand.is_dir():
            continue
        try:
            discover_mimic_files(cand)
        except FileNotFoundError:
            continue
        return cand
    raise FileNotFoundError(
        "No MIMIC-IV dataset found under data/. Unzip MIMIC-IV-Demo v2.2 "
        "into data/mimic-iv-clinical-database-demo-2.2/ and re-run."
    )


def _prepare_cohort(db_path: Path, rate_multiplier: float) -> pd.DataFrame:
    """Load the joined cohort, compress the timeline, and stress the rate."""
    cohort = load_joined_cohort(db_path)
    cohort = cohort.sort_values("intime").reset_index(drop=True)

    # Compress: PhysioNet shifts each patient ~100 years; keep gaps capped 72h.
    deltas_h = (cohort["intime"] - cohort["intime"].shift(1)).dt.total_seconds() / 3600
    deltas_h = deltas_h.fillna(0).clip(upper=72)

    # Stress: divide gaps by the multiplier (preserves order, scales density).
    deltas_h = deltas_h / float(rate_multiplier)
    anchor = pd.Timestamp("2180-01-01")
    cohort = cohort.copy()
    cohort["intime"] = anchor + pd.to_timedelta(deltas_h.cumsum(), unit="h")
    cohort["los_hours"] = cohort["los"] * 24.0
    return cohort


def _sweep_one_cell(
    cohort: pd.DataFrame, n_beds: int
) -> Dict[str, Dict[str, float]]:
    """Run FCFS + adaptive + every Threshold reserve at the given bed count.

    Returns a mapping ``policy_name → kpi_dict``.
    """
    stream = ArrivalStream.from_dataframe(cohort)

    policies: Dict[str, object] = {
        "fcfs": FCFSPolicy(n_beds=n_beds, max_board=MAX_BOARD),
        "adaptive": AdaptiveThresholdPolicy(n_beds=n_beds, max_board=MAX_BOARD),
    }
    for r in RESERVES:
        policies[f"threshold_r{r}"] = ThresholdPolicy(
            n_beds=n_beds, reserve=r, max_board=MAX_BOARD
        )

    out: Dict[str, Dict[str, float]] = {}
    for name, policy in policies.items():
        sim = MonteCarloSimulator(
            policy=policy, stream=stream,
            n_runs=N_RUNS, n_workers=N_WORKERS,
            perturb=True, seed=42,
        )
        kpis = compute_kpis(sim.run())
        out[name] = {
            "mean_utilization": kpis.mean_utilization,
            "p95_utilization": kpis.p95_utilization,
            "admit_rate": kpis.admit_rate,
            "board_rate": kpis.board_rate,
            "divert_rate": kpis.divert_rate,
            "urgent_admit_rate": kpis.urgent_admit_rate,
            "elective_admit_rate": kpis.elective_admit_rate,
        }
    return out


# ---------- plotting ----------------------------------------------------------


def _plot_heatmap(
    grid: pd.DataFrame, value_col: str, out_path: Path, title: str, cbar_label: str
) -> None:
    """Render a (rate × beds) heatmap of `value_col`."""
    import matplotlib.pyplot as plt

    pivot = grid.pivot_table(
        index="rate_multiplier", columns="n_beds", values=value_col
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = float(pivot.values.max()) or 1e-9
    vmin = float(pivot.values.min())
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="magma",
        vmin=min(0.0, vmin), vmax=max(1e-3, vmax),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"×{r:g}" for r in pivot.index])
    ax.set_xlabel("ICU beds")
    ax.set_ylabel("Arrival-rate multiplier")
    ax.set_title(title)

    # Annotate every cell with its numeric value.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v < vmax * 0.55 else "black"
            ax.text(j, i, f"{v * 100:+.1f}%",
                    ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_admit_curves(grid: pd.DataFrame, out_path: Path) -> None:
    """Admit-rate vs arrival-rate, one line per policy, faceted by bed count.

    The 'scissor' between FCFS and Threshold lines as rate climbs is the
    headline image: it shows policy choice matters precisely when load is
    high.
    """
    import matplotlib.pyplot as plt

    beds = sorted(grid["n_beds"].unique())
    fig, axes = plt.subplots(
        1, len(beds), figsize=(3.2 * len(beds), 4.0), sharey=True
    )
    if len(beds) == 1:
        axes = [axes]

    palette = {
        "fcfs": "#444444",
        "adaptive": "#9467bd",
        "threshold_r1": "#1f77b4",
        "threshold_r2": "#2ca02c",
        "threshold_r3": "#d62728",
    }
    for ax, b in zip(axes, beds):
        sub = grid[grid["n_beds"] == b]
        for policy in sub["policy"].unique():
            row = sub[sub["policy"] == policy].sort_values("rate_multiplier")
            ax.plot(
                row["rate_multiplier"], row["urgent_admit_rate"] * 100,
                marker="o", label=policy, color=palette.get(policy, None),
                linewidth=2,
            )
        ax.set_title(f"{b} beds")
        ax.set_xlabel("arrival-rate ×")
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Urgent-patient admit rate (%)")
    axes[-1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(
        "Urgent-patient protection vs load — where reservation policies matter",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------- main --------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("ICU SCHEDULER — stress-sweep across (n_beds, arrival_rate)")
    print("=" * 78)

    mimic_dir = _find_mimic_dir()
    print(f"[0] MIMIC dir: {mimic_dir}")

    db = ingest_mimic_dir(mimic_dir)
    print(f"[1] SQLite cache: {db}")

    rows: List[dict] = []
    for rate in RATES_GRID:
        cohort = _prepare_cohort(db, rate)
        span_h = (cohort["intime"].max() - cohort["intime"].min()).total_seconds() / 3600
        print(f"\n[2] rate ×{rate:g}: {len(cohort)} arrivals across {span_h:.1f}h")
        for n_beds in BEDS_GRID:
            print(f"    sweeping n_beds={n_beds} ... ", end="", flush=True)
            cell = _sweep_one_cell(cohort, n_beds)
            for policy, kpi in cell.items():
                rows.append(
                    {"rate_multiplier": rate, "n_beds": n_beds,
                     "policy": policy, **kpi}
                )
            # Quick at-a-glance: how much does the best reservation policy help?
            fcfs_div = cell["fcfs"]["divert_rate"]
            best = min(cell[p]["divert_rate"] for p in cell if p != "fcfs")
            gap = fcfs_div - best
            print(f"FCFS divert={fcfs_div * 100:.1f}%, best-reserve={best * 100:.1f}%  "
                  f"(gap = {gap * 100:+.1f} pp)")

    grid = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUTPUT_DIR / "stress_grid.csv"
    grid.to_csv(grid_path, index=False)
    print(f"\n[3] Grid written to {grid_path}  ({len(grid)} cells)")

    # --- Derived gap dataframes -----------------------------------------------
    # divert_gap = fcfs divert - best fixed/adaptive reservation divert
    fcfs = grid[grid["policy"] == "fcfs"].set_index(["rate_multiplier", "n_beds"])
    candidates = grid[
        grid["policy"].str.startswith("threshold_") | (grid["policy"] == "adaptive")
    ]
    best_reserve = candidates.groupby(["rate_multiplier", "n_beds"]).agg(
        divert_rate=("divert_rate", "min"),
        urgent_admit_rate=("urgent_admit_rate", "max"),
    )
    gaps = pd.DataFrame({
        "divert_gap": fcfs["divert_rate"] - best_reserve["divert_rate"],
        "urgent_admit_gap": best_reserve["urgent_admit_rate"] - fcfs["urgent_admit_rate"],
    }).reset_index()

    gaps_path = OUTPUT_DIR / "gaps.csv"
    gaps.to_csv(gaps_path, index=False)
    print(f"[4] Gap table written to {gaps_path}")

    print("\n[5] (rate × beds) divert-rate gap (FCFS minus best reservation), pp:")
    pivot = gaps.pivot_table(
        index="rate_multiplier", columns="n_beds", values="divert_gap"
    ).sort_index(ascending=False)
    print((pivot * 100).round(1).to_string(float_format=lambda x: f"{x:+5.1f}"))

    print("\n[6] (rate × beds) urgent-admit-rate gap (best reservation minus FCFS), pp:")
    pivot_u = gaps.pivot_table(
        index="rate_multiplier", columns="n_beds", values="urgent_admit_gap"
    ).sort_index(ascending=False)
    print((pivot_u * 100).round(1).to_string(float_format=lambda x: f"{x:+5.1f}"))

    # --- Figures --------------------------------------------------------------
    figs = OUTPUT_DIR / "figures"
    _plot_heatmap(
        gaps, "divert_gap", figs / "divert_gap.png",
        title="Divert-rate gap: FCFS minus best reservation policy",
        cbar_label="Δ divert rate",
    )
    _plot_heatmap(
        gaps, "urgent_admit_gap", figs / "urgent_protection.png",
        title="Urgent-patient protection: best reservation policy minus FCFS",
        cbar_label="Δ urgent admit rate",
    )
    _plot_admit_curves(grid, figs / "admit_curves.png")
    print(f"\n[7] Figures written to {figs}/")
    print("    - divert_gap.png")
    print("    - urgent_protection.png")
    print("    - admit_curves.png")

    # --- Headline takeaways ---------------------------------------------------
    # The urgent-admit gap is the algorithmic story: reservation policies hold
    # back beds, so under load they admit more urgent patients than FCFS, even
    # at the cost of admitting fewer total. This is the tradeoff the project
    # actually exists to expose.
    max_urg_idx = gaps["urgent_admit_gap"].idxmax()
    mu = gaps.loc[max_urg_idx]

    # Look up the elective sacrifice in the same cell, to put the tradeoff in
    # context (no point pretending Threshold is strictly dominant).
    fcfs_cell = grid[(grid["policy"] == "fcfs")
                     & (grid["n_beds"] == mu["n_beds"])
                     & (grid["rate_multiplier"] == mu["rate_multiplier"])].iloc[0]
    reserve_cell = grid[((grid["policy"].str.startswith("threshold_"))
                         | (grid["policy"] == "adaptive"))
                        & (grid["n_beds"] == mu["n_beds"])
                        & (grid["rate_multiplier"] == mu["rate_multiplier"])]
    best_reserve_row = reserve_cell.loc[reserve_cell["urgent_admit_rate"].idxmax()]
    elective_drop = (fcfs_cell["elective_admit_rate"]
                     - best_reserve_row["elective_admit_rate"]) * 100

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"Best urgent-protection regime: n_beds={int(mu['n_beds'])}, "
          f"rate=×{mu['rate_multiplier']:g}, policy={best_reserve_row['policy']}")
    print(f"  → urgent-admit rate: FCFS {fcfs_cell['urgent_admit_rate'] * 100:.1f}%  "
          f"vs reservation {best_reserve_row['urgent_admit_rate'] * 100:.1f}%  "
          f"(+{mu['urgent_admit_gap'] * 100:.1f} pp)")
    print(f"  → elective trade-off: FCFS admits "
          f"{fcfs_cell['elective_admit_rate'] * 100:.1f}% electives, "
          f"reservation {best_reserve_row['elective_admit_rate'] * 100:.1f}% "
          f"(−{elective_drop:.1f} pp)")
    print("This is the clinically relevant tradeoff: under load, FCFS fills "
          "beds with whoever shows up first, while reservation policies protect "
          "capacity for emergencies at the cost of cancelled electives.")
    print("=" * 78)


if __name__ == "__main__":
    main()
