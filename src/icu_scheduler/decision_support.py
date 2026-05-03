"""Decision-report helpers for hospital-facing ICU capacity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from icu_scheduler.evaluator import ClinicalCostWeights, compare_policies
from icu_scheduler.forecasting import HistoricalArrivalForecaster
from icu_scheduler.optimization import RollingHorizonILPPolicy
from icu_scheduler.scheduler import (
    AdaptiveThresholdPolicy,
    FCFSPolicy,
    ThresholdPolicy,
)
from icu_scheduler.simulator import MonteCarloSimulator, SimulationResult
from icu_scheduler.stream import ArrivalStream

SCENARIO_KEYS = ["rate_multiplier", "n_beds"]
RECOMMENDATION_SORT = [
    "urgent_admit_rate",
    "clinical_cost_per_arrival",
    "mean_boarding_hours",
    "mean_utilization",
]


def prepare_cohort_for_simulation(
    cohort: pd.DataFrame,
    *,
    compress_timeline: bool,
    arrival_rate_multiplier: float,
) -> pd.DataFrame:
    """Return a cohort with optional MIMIC timeline compression and surge scaling."""
    if arrival_rate_multiplier <= 0:
        raise ValueError("arrival_rate_multiplier must be > 0")

    prepared = cohort.sort_values("intime").reset_index(drop=True).copy()
    if prepared.empty:
        return _ensure_los_hours(prepared)

    if compress_timeline:
        deltas_h = (prepared["intime"] - prepared["intime"].shift(1)).dt.total_seconds() / 3600
        deltas_h = deltas_h.fillna(0).clip(upper=72)
        prepared["intime"] = pd.Timestamp("2180-01-01") + pd.to_timedelta(
            deltas_h.cumsum(), unit="h"
        )

    if arrival_rate_multiplier != 1.0:
        anchor = prepared["intime"].iloc[0]
        deltas_h = (prepared["intime"] - prepared["intime"].shift(1)).dt.total_seconds() / 3600
        deltas_h = deltas_h.fillna(0) / float(arrival_rate_multiplier)
        prepared["intime"] = anchor + pd.to_timedelta(deltas_h.cumsum(), unit="h")

    return _ensure_los_hours(prepared)


def simulate_policy_scenario(
    cohort: pd.DataFrame,
    *,
    n_beds: int,
    reserves: Sequence[int],
    n_runs: int,
    n_workers: int,
    max_board: int,
    include_forecast_ilp: bool,
    forecast_max_board: int = 2,
    forecast_max_board_wait: float = 24.0,
    enable_stepdown: bool = True,
    max_transfer_remaining_hours: float = 24.0,
    rate_multiplier: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, List[SimulationResult]]]:
    """Simulate one bed/rate scenario and return metrics plus raw results."""
    eval_cohort = cohort.sort_values("intime").reset_index(drop=True).copy()
    forecaster = None
    if include_forecast_ilp:
        forecast_history, eval_cohort = _train_eval_split(eval_cohort)
        forecaster = HistoricalArrivalForecaster.from_dataframe(forecast_history)

    stream = ArrivalStream.from_dataframe(eval_cohort)
    cost_weights = ClinicalCostWeights()
    policies = build_policy_grid(
        n_beds=n_beds,
        reserves=reserves,
        max_board=max_board,
        include_forecast_ilp=include_forecast_ilp,
        forecaster=forecaster,
        forecast_max_board=forecast_max_board,
        forecast_max_board_wait=forecast_max_board_wait,
        enable_stepdown=enable_stepdown,
        max_transfer_remaining_hours=max_transfer_remaining_hours,
        cost_weights=cost_weights,
    )

    named_results: Dict[str, List[SimulationResult]] = {}
    for name, policy in policies.items():
        sim = MonteCarloSimulator(
            policy=policy,
            stream=stream,
            n_runs=n_runs,
            n_workers=n_workers,
            perturb=True,
            seed=seed,
        )
        named_results[name] = sim.run()

    metrics = compare_policies(named_results, weights=cost_weights)
    metrics.insert(0, "rate_multiplier", float(rate_multiplier))
    metrics.insert(0, "n_beds", int(n_beds))
    return metrics, named_results


def build_policy_grid(
    *,
    n_beds: int,
    reserves: Sequence[int],
    max_board: int,
    include_forecast_ilp: bool,
    forecaster: HistoricalArrivalForecaster | None = None,
    forecast_max_board: int = 2,
    forecast_max_board_wait: float = 24.0,
    enable_stepdown: bool = True,
    max_transfer_remaining_hours: float = 24.0,
    cost_weights: ClinicalCostWeights | None = None,
) -> Dict[str, object]:
    """Build the online policy set used by the decision report."""
    if n_beds <= 0:
        raise ValueError("n_beds must be positive")

    weights = cost_weights or ClinicalCostWeights()
    policies: Dict[str, object] = {
        "fcfs": FCFSPolicy(n_beds=n_beds, max_board=max_board),
    }
    for reserve in _valid_reserves(reserves, n_beds):
        policies[f"threshold_r{reserve}"] = ThresholdPolicy(
            n_beds=n_beds, reserve=reserve, max_board=max_board
        )
    policies["adaptive_threshold"] = AdaptiveThresholdPolicy(n_beds=n_beds, max_board=max_board)

    if include_forecast_ilp:
        policies["forecast_ilp"] = RollingHorizonILPPolicy(
            n_beds=n_beds,
            horizon_hours=72.0,
            urgent_benefit=weights.urgent_diverted,
            elective_benefit=weights.elective_diverted,
            max_board=forecast_max_board,
            max_board_wait_hours=forecast_max_board_wait,
            forecaster=forecaster,
            allow_step_down=enable_stepdown,
            max_transfer_remaining_hours=max_transfer_remaining_hours,
        )
    return policies


def run_decision_grid(
    cohort: pd.DataFrame,
    *,
    bed_grid: Sequence[int],
    arrival_rate_grid: Sequence[float],
    reserves: Sequence[int],
    compress_timeline: bool,
    n_runs: int,
    n_workers: int,
    max_board: int,
) -> pd.DataFrame:
    """Run the fast FCFS/threshold/adaptive sweep for heatmaps and recommendations."""
    rows: List[pd.DataFrame] = []
    for rate in arrival_rate_grid:
        prepared = prepare_cohort_for_simulation(
            cohort,
            compress_timeline=compress_timeline,
            arrival_rate_multiplier=float(rate),
        )
        for n_beds in bed_grid:
            metrics, _results = simulate_policy_scenario(
                prepared,
                n_beds=int(n_beds),
                reserves=reserves,
                n_runs=n_runs,
                n_workers=n_workers,
                max_board=max_board,
                include_forecast_ilp=False,
                rate_multiplier=float(rate),
            )
            rows.append(metrics)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def recommend_policies(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pick the hospital recommendation for each bed/rate scenario.

    Default objective:
    1. Maximize urgent/emergency admit rate.
    2. Minimize clinical cost per arrival.
    3. Minimize mean boarding hours.
    4. Maximize mean utilization.
    """
    _require_columns(metrics, [*SCENARIO_KEYS, "policy", *RECOMMENDATION_SORT])
    rows: List[dict] = []
    for (rate, n_beds), scenario in metrics.groupby(SCENARIO_KEYS, sort=True):
        ranked = scenario.sort_values(
            RECOMMENDATION_SORT,
            ascending=[False, True, True, False],
            kind="mergesort",
        )
        best = ranked.iloc[0]
        rows.append(
            {
                "rate_multiplier": float(rate),
                "n_beds": int(n_beds),
                "recommended_policy": str(best["policy"]),
                "urgent_admit_rate": float(best["urgent_admit_rate"]),
                "clinical_cost_per_arrival": float(best["clinical_cost_per_arrival"]),
                "mean_boarding_hours": float(best["mean_boarding_hours"]),
                "mean_utilization": float(best["mean_utilization"]),
                "board_rate": float(best["board_rate"]),
                "divert_rate": float(best["divert_rate"]),
                "elective_admit_rate": float(best["elective_admit_rate"]),
            }
        )
    return pd.DataFrame(rows)


def recommendation_gaps(metrics: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    """Return recommended-policy gaps relative to FCFS for each scenario."""
    _require_columns(metrics, [*SCENARIO_KEYS, "policy", "urgent_admit_rate", "divert_rate"])
    _require_columns(
        recommendations,
        [*SCENARIO_KEYS, "recommended_policy", "urgent_admit_rate", "divert_rate"],
    )

    fcfs = (
        metrics[metrics["policy"] == "fcfs"]
        .set_index(SCENARIO_KEYS)[["urgent_admit_rate", "divert_rate"]]
        .rename(
            columns={
                "urgent_admit_rate": "fcfs_urgent_admit_rate",
                "divert_rate": "fcfs_divert_rate",
            }
        )
    )
    rec = recommendations.set_index(SCENARIO_KEYS)
    joined = rec.join(fcfs, how="left").reset_index()
    joined["urgent_admit_gap"] = joined["urgent_admit_rate"] - joined["fcfs_urgent_admit_rate"]
    joined["divert_gap"] = joined["fcfs_divert_rate"] - joined["divert_rate"]
    return joined


def write_summary(
    *,
    output_path: Path,
    main_metrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    gaps: pd.DataFrame,
    main_n_beds: int,
    main_rate_multiplier: float,
) -> Path:
    """Write a short Markdown report summary for the hospital decision output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_recommendation = recommend_policies(main_metrics).iloc[0]
    best_gap = gaps.sort_values(
        ["urgent_admit_gap", "divert_gap"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]

    lines = [
        "# ICU Decision Report Summary",
        "",
        "Default objective: maximize urgent/emergency admit rate. Ties are broken by "
        "lower clinical cost, lower boarding time, and higher ICU utilization.",
        "",
        "## Main Scenario",
        "",
        (
            f"At {main_n_beds} ICU beds and arrival-rate x{main_rate_multiplier:g}, "
            f"the recommended policy is `{main_recommendation['recommended_policy']}` "
            f"with urgent admit rate {main_recommendation['urgent_admit_rate']:.1%}, "
            f"divert rate {main_recommendation['divert_rate']:.1%}, "
            f"board rate {main_recommendation['board_rate']:.1%}, and mean "
            f"utilization {main_recommendation['mean_utilization']:.1%}."
        ),
        "",
        "## Sweep Headline",
        "",
        (
            f"The largest urgent-protection gain in the bed/rate sweep occurs at "
            f"{int(best_gap['n_beds'])} beds and arrival-rate "
            f"x{best_gap['rate_multiplier']:g}: "
            f"`{best_gap['recommended_policy']}` improves urgent admit rate by "
            f"{best_gap['urgent_admit_gap'] * 100:+.1f} percentage points versus "
            "FCFS."
        ),
        "",
        "## Generated Files",
        "",
        "- `metrics.csv`: main-scenario policy metrics, including forecast ILP.",
        "- `scenario_metrics.csv`: full bed-count by arrival-rate sweep.",
        "- `recommendations.csv`: recommended policy for each scenario.",
        "- `figures/`: comparison chart, occupancy curves, heatmaps, and admit curves.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _ensure_los_hours(cohort: pd.DataFrame) -> pd.DataFrame:
    prepared = cohort.copy()
    if "los_hours" not in prepared.columns and "los" in prepared.columns:
        prepared["los_hours"] = prepared["los"] * 24.0
    return prepared


def _train_eval_split(cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(cohort) >= 20:
        train_n = max(5, int(0.30 * len(cohort)))
        return cohort.iloc[:train_n].copy(), cohort.iloc[train_n:].copy()
    return cohort.copy(), cohort.copy()


def _valid_reserves(reserves: Iterable[int], n_beds: int) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(r for r in reserves if 0 <= int(r) <= n_beds))


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
