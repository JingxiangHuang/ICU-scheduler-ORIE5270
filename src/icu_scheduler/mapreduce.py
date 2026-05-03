"""A tiny MapReduce abstraction — exercises W9 course material.

The mapper emits ``(key, value)`` pairs, the framework groups by key, and the
reducer collapses each key's value list into a single output value.

We implement the map phase with ``multiprocessing.Pool`` (W8) so the same code
both demonstrates MapReduce *and* runs in parallel.
"""

from __future__ import annotations

from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd

KV = Tuple[Any, Any]
Mapper = Callable[[Any], Iterable[KV]]
Reducer = Callable[[Any, List[Any]], Any]


def map_reduce(
    data: Iterable[Any],
    mapper: Mapper,
    reducer: Reducer,
    n_workers: int = 1,
) -> Dict[Any, Any]:
    """Run a single-round MapReduce job.

    Parameters
    ----------
    data
        Input records (any iterable).
    mapper
        Callable mapping each input record to an iterable of ``(key, value)``
        pairs. Must be picklable if ``n_workers > 1``.
    reducer
        Callable mapping ``(key, list_of_values)`` to a single output value.
    n_workers
        Number of worker processes. ``1`` runs sequentially (useful for tests
        where the mapper is a closure that can't be pickled).

    Returns
    -------
    dict
        Mapping from key to reduced value.

    Examples
    --------
    >>> def m(word): yield word, 1
    >>> def r(key, values): return sum(values)
    >>> map_reduce(["a", "b", "a", "a"], m, r)
    {'a': 3, 'b': 1}
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    data_list = list(data)
    if n_workers == 1 or len(data_list) <= 1:
        mapped: List[Iterable[KV]] = [list(mapper(item)) for item in data_list]
    else:
        with Pool(processes=n_workers) as pool:
            mapped = pool.map(_collect, [(mapper, item) for item in data_list])

    grouped: Dict[Any, List[Any]] = defaultdict(list)
    for pairs in mapped:
        for key, value in pairs:
            grouped[key].append(value)

    return {key: reducer(key, values) for key, values in grouped.items()}


def _collect(args: Tuple[Mapper, Any]) -> List[KV]:
    """Pickle-friendly wrapper so ``Pool.map`` can call the mapper."""
    mapper, item = args
    return list(mapper(item))


# ---------- Domain helper: careunit hour aggregation ---------------------------


def _careunit_mapper(row: Dict[str, Any]) -> Iterable[KV]:
    """Module-level mapper so it's picklable by ``multiprocessing.Pool``."""
    yield row["first_careunit"], float(row["los"]) * 24.0


def _sum_reducer(_key: Any, values: List[float]) -> float:
    """Module-level reducer so it's picklable by ``multiprocessing.Pool``."""
    return float(sum(values))


def aggregate_careunit_hours(
    cohort: pd.DataFrame,
    n_workers: int = 1,
) -> Dict[str, float]:
    """Total ICU hours per careunit, computed via MapReduce.

    Parameters
    ----------
    cohort
        DataFrame with columns ``first_careunit`` and ``los`` (days).
    n_workers
        Workers for the map phase.

    Returns
    -------
    dict
        ``{careunit: total_hours}``.
    """
    records = cohort[["first_careunit", "los"]].to_dict("records")
    return map_reduce(records, _careunit_mapper, _sum_reducer, n_workers=n_workers)
