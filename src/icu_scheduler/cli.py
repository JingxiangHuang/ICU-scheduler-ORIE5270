"""Command-line interface. Entry point registered in pyproject as ``icu-scheduler``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import click
from sqlalchemy import create_engine, inspect


@click.group()
@click.version_option()
def main() -> None:
    """ICU Scheduler — MIMIC-IV resource allocation toolkit."""


#: Candidate locations (relative to the current working directory) that are
#: auto-scanned for a MIMIC-IV dataset when the user runs ``icu-scheduler``
#: without any ``--db`` / ``--mimic-dir`` flag. The first folder that contains
#: ``admissions.csv[.gz]`` and ``icustays.csv[.gz]`` (flat or under ``hosp/`` +
#: ``icu/``) wins. Ordered most-specific → least-specific so a user who
#: unzipped directly into ``data/`` gets picked up before any deeper nested copy.
_DATA_SEARCH_PATHS: tuple = (
    Path("data"),
    Path("data/mimic"),
    Path("data/mimic-iv"),
    Path("data/mimic-iv-demo"),
    Path("data/mimic-iv-clinical-database-demo-2.2"),
)


def _autodiscover_mimic_dir() -> Optional[Path]:
    """Look under ``./data/`` (and common subfolders) for a MIMIC-IV dataset.

    Returns the first directory that looks like a valid MIMIC dataset, or
    ``None`` if none of the candidates contain the required CSVs.
    """
    from icu_scheduler.mimic_ingest import discover_mimic_files

    for candidate in _DATA_SEARCH_PATHS:
        if not candidate.is_dir():
            continue
        try:
            discover_mimic_files(candidate)
        except FileNotFoundError:
            # Folder exists but doesn't contain the required CSVs — skip.
            continue
        return candidate

    # Last-ditch: scan one level into ./data/ for any immediate child that looks
    # like a MIMIC dataset (handles users who unzip into a nested folder whose
    # name we didn't anticipate, e.g. ``data/mimic-iv-3.1-full``).
    data_root = Path("data")
    if data_root.is_dir():
        for child in sorted(data_root.iterdir()):
            if not child.is_dir():
                continue
            try:
                discover_mimic_files(child)
            except FileNotFoundError:
                continue
            return child

    return None


def _resolve_db(
    db_path: Optional[Path],
    mimic_dir: Optional[Path] = None,
    force_reingest: bool = False,
) -> Path:
    """Return a usable SQLite path.

    Resolution order:
    1. ``--db <path>`` — a pre-built SQLite wins if given.
    2. ``--mimic-dir <dir>`` — explicit MIMIC folder → auto-ingest.
    3. **Auto-discovery under** ``./data/`` — if the user has unzipped a MIMIC
       dataset into ``data/`` (or a common subfolder) we find it and ingest it
       with no flags required.

    If no MIMIC-shaped dataset is found under ``./data/``, fail loudly instead
    of silently falling back to sample or synthetic data.
    """
    from icu_scheduler.mimic_ingest import ingest_mimic_dir

    if db_path is not None:
        return db_path

    if mimic_dir is not None:
        click.echo(f"[icu-scheduler] Ingesting MIMIC dir: {mimic_dir}")
        cache = ingest_mimic_dir(mimic_dir, force=force_reingest)
        click.echo(f"[icu-scheduler] SQLite cache ready at {cache}")
        return cache

    # --- Auto-discover under ./data/ ---
    discovered = _autodiscover_mimic_dir()
    if discovered is not None:
        click.echo(f"[icu-scheduler] Auto-discovered MIMIC data at: {discovered}")
        cache = ingest_mimic_dir(discovered, force=force_reingest)
        click.echo(f"[icu-scheduler] SQLite cache ready at {cache}")
        return cache

    data_root = Path("data")
    if not data_root.exists():
        raise click.ClickException(
            "No ./data directory found. Download MIMIC-IV Demo, place it under "
            "./data/, or pass --mimic-dir /path/to/mimic or --db /path/to/cache.sqlite."
        )

    raise click.ClickException(
        "No MIMIC-shaped dataset found under ./data. Expected admissions and "
        "icustays CSV files, either directly under data/ or in hosp/ and icu/ "
        "subfolders. Download MIMIC-IV Demo into ./data/, or pass --mimic-dir "
        "or --db explicitly."
    )


def _parse_int_tuple(raw: str, param_name: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list from a CLI option."""
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise click.BadParameter(f"{param_name} must be comma-separated integers ({exc})") from exc
    if not values:
        raise click.BadParameter(f"{param_name} must contain at least one integer")
    return values


def _parse_float_tuple(raw: str, param_name: str) -> tuple[float, ...]:
    """Parse a comma-separated positive-float list from a CLI option."""
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise click.BadParameter(f"{param_name} must be comma-separated numbers ({exc})") from exc
    if not values:
        raise click.BadParameter(f"{param_name} must contain at least one number")
    if any(value <= 0 for value in values):
        raise click.BadParameter(f"{param_name} values must be > 0")
    return values


def _run_pipeline(
    db_path: Path,
    n_runs: int,
    n_workers: int,
    output: Path,
    n_beds: int = 20,
    reserves: tuple = (2, 5),
    max_board: int = 5,
    forecast_max_board: int = 2,
    forecast_max_board_wait: float = 24.0,
    enable_stepdown: bool = True,
    max_transfer_remaining_hours: float = 24.0,
    compress_timeline: bool = False,
    arrival_rate_multiplier: float = 1.0,
) -> Dict[str, "List"]:
    """Shared helper: load data, run all policies, write reports.

    Parameters
    ----------
    n_beds
        Number of ICU beds in the simulation. The most important tuning knob:
        too many and every policy admits everything; too few and every policy
        diverts everyone. Pick a value where policies actually differ.
    reserves
        Reservation values to sweep for ``ThresholdPolicy``. E.g. ``(1, 2, 3)``
        produces three threshold policies alongside the FCFS baseline.
    max_board
        Boarding queue capacity shared by every policy.
    forecast_max_board
        Boarding queue capacity for the forecast-driven ILP policy.
    forecast_max_board_wait
        Maximum expected hours a forecast-ILP patient can wait before boarding
        is allowed.
    enable_stepdown
        If True, forecast ILP may step down a low-acuity near-discharge ICU
        patient to admit an urgent/emergency arrival when the unit is full.
    max_transfer_remaining_hours
        Maximum remaining ICU LOS for patients eligible for step-down transfer.
    compress_timeline
        If True, collapse inter-arrival gaps onto a unified timeline (capped
        at 72h) so MIMIC-IV's 91-year deidentification spread doesn't hide
        bed contention. Strongly recommended when feeding real MIMIC-IV.
    arrival_rate_multiplier
        Divide every inter-arrival gap by this factor (after compression).
        ``2.0`` simulates a ward where patients arrive twice as fast; ``3.0``
        simulates peak / crisis conditions. ``1.0`` is historical pace.
        Lengths of stay are **not** scaled, so this is a pure arrival-rate
        stress test — the standard way to probe policy sensitivity in
        queueing simulations.
    """
    import pandas as pd
    from icu_scheduler.data_loader import load_joined_cohort
    from icu_scheduler.evaluator import ClinicalCostWeights, compare_policies
    from icu_scheduler.forecasting import HistoricalArrivalForecaster
    from icu_scheduler.mapreduce import aggregate_careunit_hours
    from icu_scheduler.optimization import RollingHorizonILPPolicy
    from icu_scheduler.plotting import plot_occupancy_bands, plot_policy_comparison
    from icu_scheduler.scheduler import AdaptiveThresholdPolicy, FCFSPolicy, ThresholdPolicy
    from icu_scheduler.simulator import MonteCarloSimulator
    from icu_scheduler.stream import ArrivalStream

    click.echo(f"[icu-scheduler] Loading cohort from {db_path}")
    cohort = load_joined_cohort(db_path)
    click.echo(f"[icu-scheduler] Cohort size: {len(cohort)} stays")

    # Optionally compress timeline: MIMIC-IV deidentifies by shifting each
    # patient's timestamps by ~100 years, so naive playback makes arrivals
    # look 91 years apart and no bed contention ever happens. We keep the
    # inter-arrival gaps (capped at 72h) and re-anchor to a unified t=0.
    if compress_timeline and len(cohort) > 0:
        cohort = cohort.sort_values("intime").reset_index(drop=True)
        deltas = (cohort["intime"] - cohort["intime"].shift(1)).dt.total_seconds() / 3600
        deltas = deltas.fillna(0).clip(upper=72)
        unified_start = pd.Timestamp("2180-01-01")
        cohort["intime"] = unified_start + pd.to_timedelta(deltas.cumsum(), unit="h")
        span_days = (cohort["intime"].max() - cohort["intime"].min()).days
        click.echo(f"[icu-scheduler] Timeline compressed → spans {span_days} days")

    # Arrival-rate stress test: divide every inter-arrival gap by the multiplier.
    # This squeezes the same patients into a shorter window without touching LOS,
    # which is the textbook way to push a queueing system toward saturation and
    # actually expose policy differences. multiplier=1.0 is a no-op.
    if arrival_rate_multiplier != 1.0 and len(cohort) > 0:
        if arrival_rate_multiplier <= 0:
            raise click.BadParameter(
                f"--arrival-rate-multiplier must be > 0 (got {arrival_rate_multiplier})"
            )
        cohort = cohort.sort_values("intime").reset_index(drop=True)
        anchor = cohort["intime"].iloc[0]
        deltas_h = (cohort["intime"] - cohort["intime"].shift(1)).dt.total_seconds() / 3600
        deltas_h = deltas_h.fillna(0) / float(arrival_rate_multiplier)
        cohort = cohort.copy()
        cohort["intime"] = anchor + pd.to_timedelta(deltas_h.cumsum(), unit="h")
        new_span_h = (cohort["intime"].max() - cohort["intime"].min()).total_seconds() / 3600
        click.echo(
            f"[icu-scheduler] Arrival-rate stress test ×{arrival_rate_multiplier:g} "
            f"→ {len(cohort)} arrivals across {new_span_h:.1f}h"
        )

    # MapReduce aggregation — a course-skill flex.
    careunit_hours = aggregate_careunit_hours(cohort, n_workers=min(n_workers, 4))
    click.echo("[icu-scheduler] Careunit hours (MapReduce):")
    for unit, hours in sorted(careunit_hours.items(), key=lambda kv: -kv[1]):
        click.echo(f"    {unit:<40}  {hours:>10.1f} hours")

    # Train a deployable forecaster on historical data, then evaluate the
    # online policies on later holdout arrivals when the cohort is large enough.
    cohort = cohort.sort_values("intime").reset_index(drop=True)
    cohort = cohort.assign(los_hours=cohort["los"] * 24.0)
    if len(cohort) >= 20:
        train_n = max(5, int(0.30 * len(cohort)))
        forecast_history = cohort.iloc[:train_n].copy()
        eval_cohort = cohort.iloc[train_n:].copy()
        click.echo(
            f"[icu-scheduler] Forecast model trained on {len(forecast_history)} "
            f"historical stays; evaluating on {len(eval_cohort)} holdout stays"
        )
    else:
        forecast_history = cohort.copy()
        eval_cohort = cohort.copy()
        click.echo("[icu-scheduler] Tiny cohort: training forecast model on all stays")

    forecaster = HistoricalArrivalForecaster.from_dataframe(forecast_history)
    stream = ArrivalStream.from_dataframe(eval_cohort)

    # Build the policy grid. FCFS is always the baseline.
    policies: Dict[str, object] = {
        "fcfs": FCFSPolicy(n_beds=n_beds, max_board=max_board),
    }
    for r in reserves:
        policies[f"threshold_r{r}"] = ThresholdPolicy(n_beds=n_beds, reserve=r, max_board=max_board)
    policies["adaptive_threshold"] = AdaptiveThresholdPolicy(n_beds=n_beds, max_board=max_board)
    cost_weights = ClinicalCostWeights()
    policies["forecast_ilp"] = RollingHorizonILPPolicy(
        n_beds=n_beds,
        horizon_hours=72.0,
        urgent_benefit=cost_weights.urgent_diverted,
        elective_benefit=cost_weights.elective_diverted,
        max_board=forecast_max_board,
        max_board_wait_hours=forecast_max_board_wait,
        forecaster=forecaster,
        allow_step_down=enable_stepdown,
        max_transfer_remaining_hours=max_transfer_remaining_hours,
    )
    click.echo(
        f"[icu-scheduler] Policy grid: {list(policies)}  "
        f"(n_beds={n_beds}, max_board={max_board}, "
        f"forecast_max_board={forecast_max_board}, "
        f"forecast_max_board_wait={forecast_max_board_wait:g}h, "
        f"stepdown={enable_stepdown}, "
        f"max_transfer_remaining={max_transfer_remaining_hours:g}h)"
    )

    named_results = {}
    for name, policy in policies.items():
        click.echo(f"[icu-scheduler] Simulating policy '{name}' ({n_runs} runs)")
        sim = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=n_runs,
            n_workers=n_workers,
            perturb=True,
            seed=42,
        )
        named_results[name] = sim.run()

    # Persist KPIs + figures.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = compare_policies(named_results, weights=cost_weights)
    metrics_path = output / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    click.echo(f"[icu-scheduler] Wrote {metrics_path}")

    comp_path = plot_policy_comparison(named_results, figures_dir / "comparison.png")
    click.echo(f"[icu-scheduler] Wrote {comp_path}")

    for name, results in named_results.items():
        band_path = plot_occupancy_bands(results, figures_dir / f"occupancy_{name}.png")
        click.echo(f"[icu-scheduler] Wrote {band_path}")

    return named_results


@main.command("run-demo")
@click.option("--n-runs", default=100, show_default=True, help="Monte-Carlo replications.")
@click.option("--n-workers", default=1, show_default=True, help="Parallel workers.")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a pre-built MIMIC-IV SQLite DB. If omitted, use --mimic-dir or "
    "auto-discover a MIMIC-shaped dataset under ./data. The command errors if "
    "no data source is found.",
)
@click.option(
    "--mimic-dir",
    "mimic_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder you unzipped MIMIC-IV (or the Demo) into. CSVs are auto-discovered "
    "(flat layout or hosp/ + icu/ subfolders; .csv or .csv.gz) and ingested into a "
    "cached SQLite file. Subsequent runs reuse the cache.",
)
@click.option(
    "--force-reingest",
    is_flag=True,
    default=False,
    help="Rebuild the SQLite cache even if the fingerprint matches.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("reports"),
    show_default=True,
    help="Output directory for figures and metrics.",
)
@click.option(
    "--n-beds",
    default=20,
    show_default=True,
    help="ICU bed count. Tune this to the point where policies actually differ: "
    "too high → all policies admit everyone; too low → all divert. For the "
    "MIMIC-IV-Demo v2.2 cohort (140 stays) use e.g. 6 with --compress-timeline.",
)
@click.option(
    "--reserves",
    default="2,5",
    show_default=True,
    help="Comma-separated list of reservation values to sweep for ThresholdPolicy. "
    "E.g. --reserves 1,2,3 produces three threshold policies.",
)
@click.option(
    "--max-board",
    default=5,
    show_default=True,
    help="Boarding queue capacity shared by every policy.",
)
@click.option(
    "--forecast-max-board",
    default=2,
    show_default=True,
    help="Boarding queue capacity for the forecast-driven ILP policy.",
)
@click.option(
    "--forecast-max-board-wait",
    default=24.0,
    show_default=True,
    help="Maximum expected boarding wait in hours for forecast-driven ILP.",
)
@click.option(
    "--enable-stepdown/--no-enable-stepdown",
    default=True,
    show_default=True,
    help="Allow forecast ILP to transfer low-acuity near-discharge ICU patients "
    "when an urgent/emergency arrival needs a bed.",
)
@click.option(
    "--max-transfer-remaining-hours",
    default=24.0,
    show_default=True,
    help="Maximum remaining ICU LOS for a patient eligible for step-down transfer.",
)
@click.option(
    "--compress-timeline/--no-compress-timeline",
    default=False,
    show_default=True,
    help="Collapse arrivals onto a unified timeline (inter-arrival gaps capped "
    "at 72h). Needed for real MIMIC-IV because PhysioNet deidentifies by "
    "shifting each patient ~100 years, which would otherwise spread 140 "
    "arrivals across 91 years and hide all bed contention.",
)
@click.option(
    "--arrival-rate-multiplier",
    type=float,
    default=1.0,
    show_default=True,
    help="Stress-test multiplier for arrival rate. 1.0 = historical pace; 2.0 "
    "doubles the rate (gaps halved); 3.0 simulates crisis surge. LOS is left "
    "unchanged, so this is a pure ρ=λ/μ stress test — the standard knob for "
    "exposing differences between scheduling policies.",
)
def run_demo(
    n_runs: int,
    n_workers: int,
    db_path: Optional[Path],
    mimic_dir: Optional[Path],
    force_reingest: bool,
    output: Path,
    n_beds: int,
    reserves: str,
    max_board: int,
    forecast_max_board: int,
    forecast_max_board_wait: float,
    enable_stepdown: bool,
    max_transfer_remaining_hours: float,
    compress_timeline: bool,
    arrival_rate_multiplier: float,
) -> None:
    """Run the end-to-end demo pipeline."""
    click.echo(f"[icu-scheduler] run-demo (n_runs={n_runs}, workers={n_workers})")
    reserves_tuple = _parse_int_tuple(reserves, "--reserves")
    resolved = _resolve_db(db_path, mimic_dir=mimic_dir, force_reingest=force_reingest)
    _run_pipeline(
        resolved,
        n_runs=n_runs,
        n_workers=n_workers,
        output=output,
        n_beds=n_beds,
        reserves=reserves_tuple,
        max_board=max_board,
        forecast_max_board=forecast_max_board,
        forecast_max_board_wait=forecast_max_board_wait,
        enable_stepdown=enable_stepdown,
        max_transfer_remaining_hours=max_transfer_remaining_hours,
        compress_timeline=compress_timeline,
        arrival_rate_multiplier=arrival_rate_multiplier,
    )
    click.echo("[icu-scheduler] Done.")


@main.command("decision-report")
@click.option("--n-runs", default=20, show_default=True, help="Monte-Carlo replications.")
@click.option("--n-workers", default=1, show_default=True, help="Parallel workers.")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a pre-built MIMIC-IV SQLite DB. If omitted, use --mimic-dir or "
    "auto-discover a MIMIC-shaped dataset under ./data. The command errors if "
    "no data source is found.",
)
@click.option(
    "--mimic-dir",
    "mimic_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder containing MIMIC-IV CSVs; ingested into a cached SQLite file.",
)
@click.option(
    "--force-reingest",
    is_flag=True,
    default=False,
    help="Rebuild the SQLite cache even if the fingerprint matches.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("reports/decision"),
    show_default=True,
    help="Output directory for hospital-facing metrics, figures, and summary.",
)
@click.option("--n-beds", default=6, show_default=True, help="Main-scenario ICU beds.")
@click.option(
    "--reserves",
    default="0,1,2,3,4,5,6",
    show_default=True,
    help="Comma-separated reservation values to sweep for ThresholdPolicy.",
)
@click.option(
    "--arrival-rate-grid",
    default="1,1.5,2,3,4",
    show_default=True,
    help="Comma-separated surge multipliers for heatmaps and recommendation sweep.",
)
@click.option(
    "--bed-grid",
    default="3,4,5,6,8",
    show_default=True,
    help="Comma-separated ICU bed counts for heatmaps and recommendation sweep.",
)
@click.option(
    "--main-arrival-rate",
    default=1.0,
    show_default=True,
    help="Arrival-rate multiplier for the main policy comparison including forecast ILP.",
)
@click.option(
    "--max-board",
    default=3,
    show_default=True,
    help="Boarding queue capacity for FCFS, threshold, and adaptive policies.",
)
@click.option(
    "--forecast-max-board",
    default=2,
    show_default=True,
    help="Boarding queue capacity for the forecast-driven ILP policy.",
)
@click.option(
    "--forecast-max-board-wait",
    default=24.0,
    show_default=True,
    help="Maximum expected boarding wait in hours for forecast-driven ILP.",
)
@click.option(
    "--enable-stepdown/--no-enable-stepdown",
    default=True,
    show_default=True,
    help="Allow forecast ILP to transfer low-acuity near-discharge ICU patients "
    "when an urgent/emergency arrival needs a bed.",
)
@click.option(
    "--max-transfer-remaining-hours",
    default=24.0,
    show_default=True,
    help="Maximum remaining ICU LOS for a patient eligible for step-down transfer.",
)
@click.option(
    "--compress-timeline/--no-compress-timeline",
    default=True,
    show_default=True,
    help="Collapse MIMIC-IV deidentified arrivals onto a unified timeline before "
    "running the main comparison and grid sweep.",
)
def decision_report(
    n_runs: int,
    n_workers: int,
    db_path: Optional[Path],
    mimic_dir: Optional[Path],
    force_reingest: bool,
    output: Path,
    n_beds: int,
    reserves: str,
    arrival_rate_grid: str,
    bed_grid: str,
    main_arrival_rate: float,
    max_board: int,
    forecast_max_board: int,
    forecast_max_board_wait: float,
    enable_stepdown: bool,
    max_transfer_remaining_hours: float,
    compress_timeline: bool,
) -> None:
    """Generate a hospital-facing ICU bed-allocation decision report."""
    from icu_scheduler.data_loader import load_joined_cohort
    from icu_scheduler.decision_support import (
        prepare_cohort_for_simulation,
        recommendation_gaps,
        recommend_policies,
        run_decision_grid,
        simulate_policy_scenario,
        write_summary,
    )
    from icu_scheduler.plotting import (
        plot_admit_curves,
        plot_metric_heatmap,
        plot_occupancy_bands,
        plot_policy_comparison,
    )

    reserves_tuple = _parse_int_tuple(reserves, "--reserves")
    arrival_rates = _parse_float_tuple(arrival_rate_grid, "--arrival-rate-grid")
    bed_counts = _parse_int_tuple(bed_grid, "--bed-grid")
    if any(count <= 0 for count in bed_counts):
        raise click.BadParameter("--bed-grid values must be positive")
    if n_beds <= 0:
        raise click.BadParameter("--n-beds must be positive")
    if main_arrival_rate <= 0:
        raise click.BadParameter("--main-arrival-rate must be > 0")

    click.echo(
        f"[icu-scheduler] decision-report "
        f"(objective=urgent protection, n_runs={n_runs}, workers={n_workers})"
    )
    resolved = _resolve_db(db_path, mimic_dir=mimic_dir, force_reingest=force_reingest)
    click.echo(f"[icu-scheduler] Loading cohort from {resolved}")
    cohort = load_joined_cohort(resolved)
    click.echo(f"[icu-scheduler] Cohort size: {len(cohort)} stays")

    output = Path(output)
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    click.echo("[icu-scheduler] Running main policy comparison including forecast_ilp")
    main_cohort = prepare_cohort_for_simulation(
        cohort,
        compress_timeline=compress_timeline,
        arrival_rate_multiplier=main_arrival_rate,
    )
    main_metrics, named_results = simulate_policy_scenario(
        main_cohort,
        n_beds=n_beds,
        reserves=reserves_tuple,
        n_runs=n_runs,
        n_workers=n_workers,
        max_board=max_board,
        include_forecast_ilp=True,
        forecast_max_board=forecast_max_board,
        forecast_max_board_wait=forecast_max_board_wait,
        enable_stepdown=enable_stepdown,
        max_transfer_remaining_hours=max_transfer_remaining_hours,
        rate_multiplier=main_arrival_rate,
    )
    metrics_path = output / "metrics.csv"
    main_metrics.to_csv(metrics_path, index=False)
    click.echo(f"[icu-scheduler] Wrote {metrics_path}")

    comparison_path = plot_policy_comparison(named_results, figures_dir / "policy_comparison.png")
    click.echo(f"[icu-scheduler] Wrote {comparison_path}")
    for name, results in named_results.items():
        occupancy_path = plot_occupancy_bands(results, figures_dir / f"occupancy_{name}.png")
        click.echo(f"[icu-scheduler] Wrote {occupancy_path}")

    click.echo(
        "[icu-scheduler] Running bed-count x arrival-rate sweep "
        "(fcfs, threshold, adaptive_threshold)"
    )
    scenario_metrics = run_decision_grid(
        cohort,
        bed_grid=bed_counts,
        arrival_rate_grid=arrival_rates,
        reserves=reserves_tuple,
        compress_timeline=compress_timeline,
        n_runs=n_runs,
        n_workers=n_workers,
        max_board=max_board,
    )
    scenario_path = output / "scenario_metrics.csv"
    scenario_metrics.to_csv(scenario_path, index=False)
    click.echo(f"[icu-scheduler] Wrote {scenario_path}")

    recommendations = recommend_policies(scenario_metrics)
    recommendations_path = output / "recommendations.csv"
    recommendations.to_csv(recommendations_path, index=False)
    click.echo(f"[icu-scheduler] Wrote {recommendations_path}")

    gaps = recommendation_gaps(scenario_metrics, recommendations)
    gaps_path = output / "gaps.csv"
    gaps.to_csv(gaps_path, index=False)
    click.echo(f"[icu-scheduler] Wrote {gaps_path}")

    urgent_heatmap = plot_metric_heatmap(
        gaps,
        "urgent_admit_gap",
        figures_dir / "urgent_protection_heatmap.png",
        title="Urgent protection: recommended policy minus FCFS",
        cbar_label="Delta urgent admit rate",
    )
    divert_heatmap = plot_metric_heatmap(
        gaps,
        "divert_gap",
        figures_dir / "divert_gap_heatmap.png",
        title="Divert-rate improvement: FCFS minus recommended policy",
        cbar_label="Delta divert rate",
    )
    curves_path = plot_admit_curves(scenario_metrics, figures_dir / "admit_curves.png")
    click.echo(f"[icu-scheduler] Wrote {urgent_heatmap}")
    click.echo(f"[icu-scheduler] Wrote {divert_heatmap}")
    click.echo(f"[icu-scheduler] Wrote {curves_path}")

    summary_path = write_summary(
        output_path=output / "summary.md",
        main_metrics=main_metrics,
        recommendations=recommendations,
        gaps=gaps,
        main_n_beds=n_beds,
        main_rate_multiplier=main_arrival_rate,
    )
    click.echo(f"[icu-scheduler] Wrote {summary_path}")
    click.echo("[icu-scheduler] Decision report complete.")


@main.command("optimize")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a pre-built MIMIC-IV SQLite DB. If omitted, use --mimic-dir or "
    "auto-discover a MIMIC-shaped dataset under ./data. The command errors if "
    "no data source is found.",
)
@click.option(
    "--mimic-dir",
    "mimic_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder containing MIMIC-IV CSVs; ingested into a cached SQLite file.",
)
@click.option(
    "--force-reingest",
    is_flag=True,
    default=False,
    help="Rebuild the SQLite cache even if the fingerprint matches.",
)
@click.option("--n-beds", default=6, show_default=True, help="ICU bed capacity.")
@click.option(
    "--urgent-benefit",
    default=100.0,
    show_default=True,
    help="Objective benefit for admitting emergency/urgent patients.",
)
@click.option(
    "--elective-benefit",
    default=20.0,
    show_default=True,
    help="Objective benefit for admitting elective/other patients.",
)
@click.option(
    "--compress-timeline/--no-compress-timeline",
    default=True,
    show_default=True,
    help="Compress MIMIC-IV deidentified timelines before optimization.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("reports/optimization"),
    show_default=True,
    help="Output directory for ILP decisions and summary CSVs.",
)
def optimize(
    db_path: Optional[Path],
    mimic_dir: Optional[Path],
    force_reingest: bool,
    n_beds: int,
    urgent_benefit: float,
    elective_benefit: float,
    compress_timeline: bool,
    output: Path,
) -> None:
    """Solve the offline ILP bed-allocation benchmark."""
    import pandas as pd

    from icu_scheduler.data_loader import load_joined_cohort
    from icu_scheduler.optimization import optimize_admissions
    from icu_scheduler.stream import ArrivalStream

    resolved = _resolve_db(db_path, mimic_dir=mimic_dir, force_reingest=force_reingest)
    click.echo(f"[icu-scheduler] Solving ILP optimizer from {resolved}")
    cohort = load_joined_cohort(resolved)

    if compress_timeline and len(cohort) > 0:
        cohort = cohort.sort_values("intime").reset_index(drop=True)
        deltas = (cohort["intime"] - cohort["intime"].shift(1)).dt.total_seconds() / 3600
        deltas = deltas.fillna(0).clip(upper=72)
        cohort["intime"] = pd.Timestamp("2180-01-01") + pd.to_timedelta(deltas.cumsum(), unit="h")
        click.echo("[icu-scheduler] Timeline compressed for optimization")

    cohort = cohort.assign(los_hours=cohort["los"] * 24.0)
    stream = ArrivalStream.from_dataframe(cohort)
    result = optimize_admissions(
        stream,
        n_beds=n_beds,
        urgent_benefit=urgent_benefit,
        elective_benefit=elective_benefit,
    )

    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "decisions.csv"
    summary_path = output / "summary.csv"
    result.to_dataframe().to_csv(decisions_path, index=False)

    occupancy = result.occupancy_series()
    summary = pd.DataFrame(
        [
            {
                "objective_value": result.objective_value,
                "n_beds": result.n_beds,
                "n_admitted": result.n_admitted,
                "n_diverted": result.n_diverted,
                "urgent_admit_rate": result.urgent_admit_rate,
                "elective_admit_rate": result.elective_admit_rate,
                "peak_occupancy": int(occupancy.max()) if not occupancy.empty else 0,
                "solver_status": result.solver_status,
                "solver_message": result.solver_message,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    click.echo(f"[icu-scheduler] Objective value: {result.objective_value:.1f}")
    click.echo(
        f"[icu-scheduler] Admitted {result.n_admitted}, diverted {result.n_diverted}; "
        f"urgent admit={result.urgent_admit_rate:.1%}, "
        f"elective admit={result.elective_admit_rate:.1%}"
    )
    click.echo(f"[icu-scheduler] Wrote {decisions_path}")
    click.echo(f"[icu-scheduler] Wrote {summary_path}")


@main.command("ingest-mimic")
@click.argument("mimic_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "cache_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the SQLite cache. Defaults to " "<mimic-dir>/.icu_scheduler_cache.sqlite.",
)
@click.option("--force", is_flag=True, default=False, help="Rebuild even if cached.")
def ingest_mimic(mimic_dir: Path, cache_path: Optional[Path], force: bool) -> None:
    """Ingest MIMIC-IV CSVs into a SQLite cache without running the simulator."""
    from icu_scheduler.mimic_ingest import ingest_mimic_dir

    out = ingest_mimic_dir(mimic_dir, cache_path=cache_path, force=force)
    click.echo(f"[icu-scheduler] Cache ready at {out}")


@main.command("validate-db")
@click.argument("sqlite_path", type=click.Path(exists=True, path_type=Path))
def validate_db(sqlite_path: Path) -> None:
    """Check that a MIMIC-IV SQLite file has the expected tables and columns."""
    expected_tables = {"admissions", "icustays"}
    expected_cols = {
        "admissions": {"subject_id", "hadm_id", "admittime", "admission_type"},
        "icustays": {"subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"},
    }

    engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    missing_tables = expected_tables - tables
    if missing_tables:
        raise click.ClickException(f"Missing tables: {missing_tables}")

    for table, cols in expected_cols.items():
        actual_cols = {c["name"] for c in insp.get_columns(table)}
        missing_cols = cols - actual_cols
        if missing_cols:
            raise click.ClickException(f"Table '{table}' missing columns: {missing_cols}")

    click.echo(f"[icu-scheduler] {sqlite_path} is valid.")


if __name__ == "__main__":
    main()
