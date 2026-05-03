"""KPI computation from simulation results.

Pure functions — the easiest things to unit-test in the whole project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from icu_scheduler.simulator import SimulationResult


@dataclass(frozen=True)
class ClinicalCostWeights:
    """Weights for the ICU allocation objective.

    Defaults encode the course-project thesis: urgent diversion is the worst
    failure mode, urgent boarding is also serious, elective cancellation is
    costly but less severe, and unused beds carry a small opportunity cost.
    """

    urgent_diverted: float = 100.0
    urgent_boarded: float = 30.0
    elective_diverted: float = 20.0
    elective_boarded: float = 10.0
    boarding_hour: float = 0.25
    transfer: float = 8.0
    idle_bed_step: float = 1.0


@dataclass(frozen=True)
class KPISummary:
    """Aggregate metrics across Monte-Carlo runs."""

    mean_utilization: float
    p95_utilization: float
    admit_rate: float
    board_rate: float
    divert_rate: float
    max_occupancy: float
    urgent_admit_rate: float
    elective_admit_rate: float
    mean_boarding_hours: float
    transfer_rate: float
    clinical_cost_per_arrival: float


def simulation_cost(
    result: SimulationResult,
    weights: ClinicalCostWeights = ClinicalCostWeights(),
) -> float:
    """Return weighted clinical/operational cost for one simulation run."""
    idle_steps = (
        float(np.sum(1.0 - result.utilization_series))
        if result.utilization_series.size > 0
        else 0.0
    )
    return (
        weights.urgent_diverted * result.n_urgent_diverted
        + weights.urgent_boarded * result.n_urgent_boarded
        + weights.elective_diverted * result.n_elective_diverted
        + weights.elective_boarded * result.n_elective_boarded
        + weights.boarding_hour * result.total_boarding_hours
        + weights.transfer * result.n_transferred
        + weights.idle_bed_step * idle_steps
    )


def compute_kpis(
    results: Iterable[SimulationResult],
    weights: ClinicalCostWeights = ClinicalCostWeights(),
) -> KPISummary:
    """Aggregate KPIs across many simulation runs.

    Parameters
    ----------
    results
        Iterable of :class:`SimulationResult`.

    Returns
    -------
    KPISummary

    Raises
    ------
    ValueError
        If ``results`` is empty.
    """
    results_list = list(results)
    if not results_list:
        raise ValueError("results must be non-empty")

    total = sum(r.n_admitted + r.n_boarded + r.n_diverted for r in results_list)
    # Avoid division-by-zero on empty simulations.
    denom = float(total) if total > 0 else 1.0

    total_urgent = sum(r.n_urgent_arrivals for r in results_list)
    total_elective = sum(r.n_elective_arrivals for r in results_list)
    denom_urgent = float(total_urgent) if total_urgent > 0 else 1.0
    denom_elective = float(total_elective) if total_elective > 0 else 1.0

    return KPISummary(
        mean_utilization=float(np.mean([r.mean_utilization for r in results_list])),
        p95_utilization=float(
            np.percentile(
                [
                    float(r.utilization_series.max()) if r.utilization_series.size > 0 else 0.0
                    for r in results_list
                ],
                95,
            )
        ),
        admit_rate=sum(r.n_admitted for r in results_list) / denom,
        board_rate=sum(r.n_boarded for r in results_list) / denom,
        divert_rate=sum(r.n_diverted for r in results_list) / denom,
        max_occupancy=float(np.mean([r.max_occupancy for r in results_list])),
        urgent_admit_rate=sum(r.n_urgent_admitted for r in results_list) / denom_urgent,
        elective_admit_rate=sum(r.n_elective_admitted for r in results_list) / denom_elective,
        mean_boarding_hours=float(np.mean([r.mean_boarding_hours for r in results_list])),
        transfer_rate=sum(r.n_transferred for r in results_list) / denom,
        clinical_cost_per_arrival=sum(simulation_cost(r, weights) for r in results_list) / denom,
    )


def compare_policies(
    named_results: Dict[str, List[SimulationResult]],
    weights: ClinicalCostWeights = ClinicalCostWeights(),
) -> pd.DataFrame:
    """Compute KPIs for each policy and return a tidy DataFrame.

    Parameters
    ----------
    named_results
        Mapping from policy name → list of :class:`SimulationResult`.

    Returns
    -------
    pd.DataFrame
        One row per policy, columns = KPISummary fields plus ``policy``.
    """
    rows = []
    for policy_name, results in named_results.items():
        summary = compute_kpis(results, weights=weights)
        row = {"policy": policy_name, **asdict(summary)}
        rows.append(row)
    return pd.DataFrame(rows)
