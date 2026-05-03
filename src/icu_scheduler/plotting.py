"""Plotting helpers. Kept thin; all heavy logic lives elsewhere.

We separate plotting from the simulator so tests for the simulator don't have
to depend on matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / grader machines
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from icu_scheduler.evaluator import compute_kpis  # noqa: E402
from icu_scheduler.simulator import SimulationResult  # noqa: E402


def _align_series(results: List[SimulationResult]) -> np.ndarray:
    """Stack per-run utilization series, truncating to the shortest length."""
    if not results:
        return np.zeros((0, 0))
    min_len = min(r.utilization_series.size for r in results)
    if min_len == 0:
        return np.zeros((len(results), 0))
    return np.vstack([r.utilization_series[:min_len] for r in results])


def plot_occupancy_bands(
    results: List[SimulationResult],
    output_path: Path,
) -> Path:
    """Plot mean occupancy with 5/95 percentile bands across runs.

    Parameters
    ----------
    results
        Per-run simulation results.
    output_path
        Destination ``.png`` path. Parent directory is created if missing.

    Returns
    -------
    pathlib.Path
        The ``output_path`` that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stacked = _align_series(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    if stacked.size > 0:
        x = np.arange(stacked.shape[1])
        ax.plot(x, stacked.mean(axis=0), label="mean utilization", color="C0")
        ax.fill_between(
            x,
            np.percentile(stacked, 5, axis=0),
            np.percentile(stacked, 95, axis=0),
            alpha=0.25,
            color="C0",
            label="5–95%",
        )
    ax.set_xlabel("Arrival index")
    ax.set_ylabel("ICU utilization")
    ax.set_ylim(0, 1.05)
    ax.set_title("ICU occupancy across Monte-Carlo runs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_policy_comparison(
    named_results: Dict[str, List[SimulationResult]],
    output_path: Path,
) -> Path:
    """Bar chart of admit/board/divert rates per policy.

    Parameters
    ----------
    named_results
        Mapping policy_name → list of :class:`SimulationResult`.
    output_path
        Destination ``.png``.

    Returns
    -------
    pathlib.Path
        The ``output_path`` that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(named_results.keys())
    kpis = [compute_kpis(named_results[n]) for n in names]
    admit = [k.admit_rate for k in kpis]
    board = [k.board_rate for k in kpis]
    divert = [k.divert_rate for k in kpis]
    transfer = [k.transfer_rate for k in kpis]
    urgent_admit = [k.urgent_admit_rate for k in kpis]
    elective_admit = [k.elective_admit_rate for k in kpis]

    x = np.arange(len(names))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 2.5 * width, admit, width, label="admit (overall)", color="C2")
    ax.bar(x - 1.5 * width, board, width, label="board", color="C1")
    ax.bar(x - 0.5 * width, divert, width, label="divert", color="C3")
    ax.bar(x + 0.5 * width, transfer, width, label="transfer", color="C6")
    ax.bar(x + 1.5 * width, urgent_admit, width, label="urgent admit", color="C4")
    ax.bar(x + 2.5 * width, elective_admit, width, label="elective admit", color="C5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Policy comparison: rates including urgent vs elective")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_metric_heatmap(
    grid: "pd.DataFrame",
    value_col: str,
    output_path: Path,
    *,
    title: str,
    cbar_label: str,
) -> Path:
    """Plot a bed-count by arrival-rate heatmap for a scenario metric."""
    import pandas as pd  # local import keeps pandas optional for plotting import

    if not isinstance(grid, pd.DataFrame):
        raise TypeError("grid must be a pandas DataFrame")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = grid.pivot_table(
        index="rate_multiplier", columns="n_beds", values=value_col
    ).sort_index(ascending=False)
    values = pivot.values
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1e-9
    vmin = float(finite.min()) if finite.size else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="magma",
        vmin=min(0.0, vmin),
        vmax=max(1e-3, vmax),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"x{r:g}" for r in pivot.index])
    ax.set_xlabel("ICU beds")
    ax.set_ylabel("Arrival-rate multiplier")
    ax.set_title(title)

    threshold = vmax * 0.55
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                label = "n/a"
                color = "white"
            else:
                label = f"{value * 100:+.1f}%"
                color = "white" if value < threshold else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_admit_curves(
    grid: "pd.DataFrame",
    output_path: Path,
    *,
    metric_col: str = "urgent_admit_rate",
) -> Path:
    """Plot urgent-admit curves by arrival-rate multiplier, faceted by bed count."""
    import pandas as pd  # local import keeps pandas optional for plotting import

    if not isinstance(grid, pd.DataFrame):
        raise TypeError("grid must be a pandas DataFrame")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beds = sorted(grid["n_beds"].unique())
    fig, axes = plt.subplots(
        1,
        len(beds),
        figsize=(max(3.4 * len(beds), 5.0), 4.0),
        sharey=True,
    )
    if len(beds) == 1:
        axes = [axes]

    for ax, n_beds in zip(axes, beds):
        sub = grid[grid["n_beds"] == n_beds]
        for policy in sub["policy"].unique():
            row = sub[sub["policy"] == policy].sort_values("rate_multiplier")
            ax.plot(
                row["rate_multiplier"],
                row[metric_col] * 100,
                marker="o",
                linewidth=2,
                label=policy,
            )
        ax.set_title(f"{n_beds:g} beds")
        ax.set_xlabel("arrival-rate x")
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Urgent/emergency admit rate (%)")
    axes[-1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Urgent protection vs arrival pressure", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path
