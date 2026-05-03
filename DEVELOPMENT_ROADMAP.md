# Development Roadmap — ICU Scheduler (Two-Week Plan)

> From today (2026-04-20) to the deadline (2026-05-05 23:59), there are 15 days in total.
> The project skeleton is already in place: **53 tests**, **11 modules/scripts**, **pyproject/CI/Sphinx all set up**.
> All you need to do is replace each `raise NotImplementedError(...)` with an actual implementation.

---

## Current Project Status Overview

```
icu_scheduler_project/
├── pyproject.toml              ✅ Package definition (with pytest/coverage config)
├── README.md                   ✅ Complete README
├── Makefile                    ✅ One-liner make test/lint/demo
├── LICENSE                     ✅ MIT
├── .gitignore                  ✅
├── .github/workflows/ci.yml    ✅ GitHub Actions for 3 Python versions × lint + pytest
├── src/icu_scheduler/
│   ├── __init__.py             ✅ Public API exports
│   ├── data_loader.py          🟡 Placeholder (SQL queries written, function bodies need filling)
│   ├── stream.py               🟡 Placeholder
│   ├── scheduler.py            🟡 Placeholder (ICUState class is complete)
│   ├── simulator.py            🟡 Placeholder (run() parallel dispatch is complete)
│   ├── evaluator.py            🟡 Placeholder
│   ├── plotting.py             🟡 Placeholder
│   └── cli.py                  🟡 Placeholder (Click framework is set up)
├── tests/                      ✅ 53 tests all written
├── scripts/build_sample_db.py  ✅ Complete
└── docs/                       ✅ Sphinx configured
```

**Grading Criteria Checklist:**
| Criterion | Requirement | Current Status |
|---|---|---|
| Python package | ✅ Required | Skeleton meets the bar |
| README | ✅ Required (4 points) | All 4 points covered |
| Unit tests >80% | ✅ Required | 53 tests written; should reach 85% once implementations are filled in |
| Clean file structure | ✅ Required | `src/`, `tests/`, `docs/` cleanly separated |
| Detailed documentation | ✅ Required | Google-style docstrings + Sphinx |
| Real dataset | ✅ Required | MIMIC-IV-Demo |
| Course skills | Bonus | SQL/streams/multiproc/MapReduce/Web scraper can be showcased |

---

## Week 1 (4/21 - 4/27): Data Layer + Core Algorithms

### Day 1 (Mon 4/21) — Environment + Git + Data Download

1. **Set up local environment**
   ```bash
   cd icu_scheduler_project
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   pytest --collect-only  # Should show 53 tests
   pytest                 # All should fail but not crash — each reports NotImplementedError
   ```
2. **Initialize Git + push to GitHub**
   ```bash
   git init -b main
   git add .
   git commit -m "scaffold: initial project structure"
   git remote add origin git@github.com:yourname/icu_scheduler.git
   git push -u origin main
   ```
3. **Download MIMIC-IV-Demo v2.2**
   - <https://physionet.org/content/mimic-iv-demo/2.2/>
   - Unzip to `data/raw/mimic-iv-demo-2.2/`
4. **Build the SQLite sample database (one-time)**
   ```bash
   python scripts/build_sample_db.py --raw data/raw/mimic-iv-demo-2.2
   # Generates data/sample/mimic_demo.sqlite (~5 MB, commit to repo)
   ```

### Day 2-3 (Tue-Wed 4/22-23) — Fill in `data_loader.py`

Goal: Pass all 9 tests in `test_data_loader.py`.

Implementation approach for each `NotImplementedError`:

- **`LoaderConfig.to_url()`**: 3-line if-elif-raise.
- **`_make_engine()`**: Already written, no changes needed.
- **`load_icu_stays()`**:
  ```python
  engine = _make_engine(source)
  with engine.connect() as conn:
      df = pd.read_sql(text(ICU_STAYS_QUERY), conn,
                       parse_dates=["intime", "outtime"])
  df = df.dropna(subset=["los"]).reset_index(drop=True)
  return df
  ```
- **`load_admissions()`**: Same pattern, using `ADMISSIONS_QUERY`.
- **`load_joined_cohort()`**: Write a SQL JOIN (**do not use pandas merge** — demonstrate SQL skills):
  ```sql
  SELECT a.*, s.stay_id, s.first_careunit, s.intime, s.outtime, s.los
  FROM admissions a
  JOIN icustays s ON a.hadm_id = s.hadm_id
  WHERE s.los * 24 >= :min_los_hours
  ORDER BY a.admittime
  ```

**Verification:** `pytest tests/test_data_loader.py -v` should be all green.

### Day 4 (Thu 4/24) — Fill in `stream.py`

Goal: Pass all 14 tests in `test_stream.py`.

- **`ArrivalStream.from_dataframe()`**:
  ```python
  events = []
  for _, row in df.iterrows():
      acuity = row.get("admission_type", "unknown").lower()
      events.append(ArrivalEvent(
          subject_id=int(row["subject_id"]),
          arrival_time=row["intime"],
          los_hours=float(row["los"]) * 24,  # los is in days, convert to hours
          acuity=acuity,
      ))
  return cls(events)
  ```
- **`reservoir_sample()`** — Algorithm R:
  ```python
  if k < 0: raise ValueError("k must be non-negative")
  rng = rng or random
  reservoir = []
  for i, item in enumerate(stream):
      if i < k:
          reservoir.append(item)
      else:
          j = rng.randint(0, i)
          if j < k:
              reservoir[j] = item
  return reservoir
  ```
- **`CountMinSketch.add()` / `estimate()`**:
  ```python
  def add(self, item, count=1):
      for row, col in enumerate(self._hashes(item)):
          self._counts[row, col] += count
  def estimate(self, item):
      return int(min(self._counts[row, col] for row, col in enumerate(self._hashes(item))))
  ```

### Day 5 (Fri 4/25) — Fill in `scheduler.py`

Goal: Pass all 11 tests in `test_scheduler.py`. This is pure logic and the easiest part to test.

- **`FCFSPolicy.decide()`**:
  ```python
  if state.free > 0: return AdmissionDecision.ADMIT
  if state.board_queue_size < self.max_board: return AdmissionDecision.BOARD
  return AdmissionDecision.DIVERT
  ```
- **`ThresholdPolicy.decide()`**:
  ```python
  is_emergency = event.acuity.lower() in ("emergency", "urgent")
  if state.free == 0:
      return AdmissionDecision.BOARD if state.board_queue_size < self.max_board else AdmissionDecision.DIVERT
  if is_emergency or state.free > self.reserve:
      return AdmissionDecision.ADMIT
  return AdmissionDecision.DIVERT
  ```
- **`DischargeHeap.pop_due()`**:
  ```python
  due = []
  while self._heap and self._heap[0].discharge_time <= now:
      due.append(heapq.heappop(self._heap))
  return due
  ```

### Day 6-7 (Sat-Sun 4/26-27) — Fill in `simulator.py`

This is the most challenging part. The core is `run_once()`:

```python
def run_once(self, run_id: int) -> SimulationResult:
    rng = np.random.default_rng((self.seed or 0) + run_id)
    events = list(self.stream)
    if self.perturb:
        idx = rng.integers(0, len(events), size=len(events))
        events = [events[i] for i in idx]
        events.sort(key=lambda e: e.arrival_time)

    heap = DischargeHeap()
    occupied = 0
    n_admitted = n_boarded = n_diverted = 0
    max_occ = 0
    utilization = []

    for ev in events:
        t_hours = (ev.arrival_time - events[0].arrival_time).total_seconds() / 3600
        for _ in heap.pop_due(t_hours):
            occupied -= 1
        state = ICUState(
            n_beds=getattr(self.policy, "n_beds", 20),
            occupied=occupied,
            board_queue_size=0,
        )
        decision = self.policy.decide(ev, state)
        if decision == AdmissionDecision.ADMIT:
            occupied += 1
            heap.push(t_hours + ev.los_hours, ev.subject_id)
            n_admitted += 1
        elif decision == AdmissionDecision.BOARD:
            n_boarded += 1
        else:
            n_diverted += 1
        max_occ = max(max_occ, occupied)
        utilization.append(occupied / state.n_beds)

    return SimulationResult(
        n_admitted=n_admitted,
        n_boarded=n_boarded,
        n_diverted=n_diverted,
        max_occupancy=max_occ,
        utilization_series=np.array(utilization),
    )
```

**Note:** The multiprocessing branch is already written (`Pool.map` in `run()`), no changes needed.

---

## Week 2 (4/28 - 5/5): Wiring It Together + Docs + Submission

### Day 8 (Mon 4/28) — Fill in `evaluator.py` + `plotting.py`

- **`compute_kpis()`**:
  ```python
  results = list(results)
  if not results: raise ValueError("empty results")
  total = sum(r.n_admitted + r.n_boarded + r.n_diverted for r in results)
  return KPISummary(
      mean_utilization=np.mean([r.mean_utilization for r in results]),
      p95_utilization=np.percentile([r.utilization_series.max() for r in results], 95),
      admit_rate=sum(r.n_admitted for r in results) / total,
      board_rate=sum(r.n_boarded for r in results) / total,
      divert_rate=sum(r.n_diverted for r in results) / total,
      max_occupancy=np.mean([r.max_occupancy for r in results]),
  )
  ```
- **`compare_policies()`**: Call `compute_kpis` for each policy and tidy into a DataFrame.

### Day 9 (Tue 4/29) — Fill in `cli.py`

Make `icu-scheduler run-demo` actually work:
1. Call `load_joined_cohort` to read `data/sample/mimic_demo.sqlite`
2. Build `ArrivalStream`
3. Three policies: `FCFSPolicy()`, `ThresholdPolicy(n_beds=20, reserve=2)`, `ThresholdPolicy(n_beds=20, reserve=5)`
4. Run each with `MonteCarloSimulator(..., n_workers=n_workers)`
5. `compare_policies()` → `reports/metrics.csv`
6. `plot_policy_comparison()` → `reports/figures/comparison.png`

### Day 10 (Wed 4/30) — Improve Coverage + Add CLI Tests

```bash
pytest --cov-report=html
open htmlcov/index.html
```

Go through red (uncovered) lines and add tests. Typical findings:
- `cli.py` not tested → add `tests/test_cli.py` using `click.testing.CliRunner`
- `plotting.py` not tested → add a smoke test that calls functions and verifies file creation
- Some `raise ValueError` branches not tested → add single-line parametrize tests

Goal: coverage ≥ 85%.

### Day 11 (Thu 5/1) — Add MapReduce Module (Course Skill Bonus)

Create `src/icu_scheduler/mapreduce.py`:

```python
"""MapReduce aggregation — exercises W9 course material."""
from collections import defaultdict
from multiprocessing import Pool
from typing import Callable, Iterable, Dict, List

def map_reduce(data, mapper, reducer, n_workers=4):
    with Pool(n_workers) as pool:
        pairs = pool.map(mapper, data)
    grouped = defaultdict(list)
    for k, v in (p for sub in pairs for p in sub):
        grouped[k].append(v)
    return {k: reducer(k, vs) for k, vs in grouped.items()}
```

Use it to compute "total ICU-hours per careunit" — add this to the README to directly claim "This project applies MapReduce."

### Day 12 (Fri 5/2) — Sphinx Documentation + Polish Docstrings

```bash
make docs
open docs/_build/html/index.html
```

Check that every public function has a Google-style docstring (Parameters / Returns / Raises / Examples).

### Day 13 (Sat 5/3) — End-to-End Smoke Test + README Screenshots

```bash
make clean
pip uninstall icu-scheduler
pip install -e ".[dev]"
make test
make demo
```

Add `reports/figures/comparison.png` as a "Results" section in the README.

### Day 14 (Sun 5/4) — Chat Labels (Course AI Policy)

The professor's slides explicitly state: ChatGPT/Claude chat logs must be exported, labeled, and uploaded to Gradescope.
Export your AI conversations today and label them using the course-provided chat labeling tool.

### Day 15 (Mon 5/5) — Final Checklist + Submission

```
[ ] make test → all green, coverage ≥ 85%
[ ] make lint → clean
[ ] make docs → builds without warnings
[ ] make demo → generates reports/figures/*.png
[ ] Every command in the README actually works
[ ] GitHub repo is public (or TA added as collaborator)
[ ] CI Actions badge is green
[ ] AI conversation logs uploaded to Gradescope
```

---

## Common Pitfalls Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| `pytest` can't find `icu_scheduler` | Forgot `pip install -e .` | `make install` |
| Coverage reports `cli.py` at 0% | pyproject omits it | Add `test_cli.py` then remove the omit rule |
| GitHub Actions fails on flake8 | E203/W503 too strict | `.github/workflows/ci.yml` already has `extend-ignore` |
| MIMIC-IV download stuck | PhysioNet requires login | The Demo version doesn't require credentials |
| `multiprocessing` hangs on Windows | Missing `if __name__ == "__main__"` | `cli.py` entry point already handles this |

---

## Course Skills Self-Check — Make sure every row in README Section 7 has code backing it

- [x] **W3 Git**: Commit history + PR workflow → GitHub commit history
- [x] **W4 TDD**: Tests written before implementation → 53 tests written first
- [x] **W6 Data Structures**: Priority queue → `scheduler.DischargeHeap`
- [x] **W7 Data Streams**: Reservoir sampling + CMS → `stream.py`
- [x] **W8 Multiprocessing**: `Pool.map` → `simulator.MonteCarloSimulator.run`
- [x] **W9 MapReduce**: Added on Day 11 → `mapreduce.py`
- [x] **W9-10 SQL**: JOIN + parameterized queries → `data_loader.load_joined_cohort`
- [ ] **W11 Web Scraping** (optional bonus): Scrape a public hospital bed capacity table into `data/raw/`

---

Good luck. The skeleton already handles the most painful engineering setup. What remains is filling in the algorithmic "meat" — 2-3 hours per day over two weeks is absolutely sufficient.
