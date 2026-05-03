# ICU Scheduler: MIMIC-IV Resource Allocation Toolkit

A reproducible Python package for ICU bed-allocation simulation and
optimization using the MIMIC-IV clinical dataset. This was developed as the
final project for ORIE 5270/6125: Big Data Technologies.

## 1. Project Purpose

Intensive Care Unit (ICU) beds are scarce. A naive first-come-first-served
policy can fill beds with lower-acuity patients and leave later urgent patients
waiting, boarded, or diverted. This project studies that operational tradeoff
with a reproducible simulation and optimization pipeline.

The project currently supports:

- SQL ingestion of MIMIC-IV admissions and ICU stay records.
- Event-stream modeling of ICU arrivals and length of stay.
- Monte-Carlo simulation of ICU occupancy under different allocation policies.
- FCFS, fixed threshold-reservation, and adaptive threshold baselines.
- A forecast-driven rolling-horizon ILP policy.
- Finite boarding with an explicit maximum waiting time.
- Conservative step-down transfer for low-acuity near-discharge ICU patients.
- Hospital-facing decision reports with recommendations, heatmaps, comparison
  charts, and occupancy curves.
- Unit tests with coverage above the course target.

The main modeling goal is to reduce clinically costly outcomes, especially
urgent diversion and long boarding, while keeping ICU use realistic.

## 2. Dataset

The current experiments use the MIMIC-IV Clinical Database Demo v2.2 from
PhysioNet:

<https://physionet.org/content/mimic-iv-demo/2.2/>

Tables used:

- `hosp.admissions`
- `icu.icustays`
- `icu.transfers`

In the local run used for the current results, the CLI auto-discovered:

```text
data/mimic-iv-clinical-database-demo-2.2/
```

After joining admissions and ICU stays, the usable cohort contained 140 ICU
stays. For the forecast-driven policy, the pipeline sorts stays
chronologically, trains the historical forecaster on the first 42 stays, and
evaluates policies on the remaining 98 holdout stays.

Because MIMIC-IV timestamps are deidentified and shifted across patients, the
demo experiments use `--compress-timeline` to preserve arrival order while
removing unrealistic multi-year gaps. Without this, the ICU almost never fills
and allocation policies become indistinguishable.

By default, the CLI only auto-discovers MIMIC-shaped datasets under `data/`.
If `data/` is missing or does not contain the required MIMIC files, commands
fail with an explicit error instead of silently using sample or synthetic data.
This keeps reproduction honest: real-data commands should either find MIMIC
under `data/`, receive an explicit `--mimic-dir`, or receive an explicit `--db`.

Data-use note: raw MIMIC files, generated SQLite caches, and generated reports
are intentionally not committed to GitHub. To reproduce the real-data
experiments, download the MIMIC-IV Demo from PhysioNet and place it locally
under `data/` as shown below. The repository keeps only instructions and a
small safe sample database for tests and explicit examples.

## 3. Installation

```bash
git clone <your-github-repo-url>
cd icu_scheduler_project
pip install -e ".[dev]"
```

Python 3.10 or newer is required.

To verify the installation:

```bash
pytest --cov=icu_scheduler --cov-report=term-missing
```

Current local verification:

```text
137 passed
90% total coverage
```

## 4. Reproducing the Decision Report

Place MIMIC-IV Demo v2.2 under `data/`:

```text
data/
  mimic-iv-clinical-database-demo-2.2/
    hosp/admissions.csv.gz
    icu/icustays.csv.gz
```

Then run:

```bash
icu-scheduler decision-report \
    --n-beds 6 \
    --reserves 0,1,2,3,4,5,6 \
    --arrival-rate-grid 1,1.5,2,3,4 \
    --bed-grid 3,4,5,6,8 \
    --compress-timeline \
    --main-arrival-rate 1 \
    --forecast-max-board 2 \
    --forecast-max-board-wait 24 \
    --max-transfer-remaining-hours 24 \
    --n-runs 20 \
    --n-workers 1 \
    --output reports/decision
```

The command writes:

- `reports/decision/metrics.csv`
- `reports/decision/scenario_metrics.csv`
- `reports/decision/recommendations.csv`
- `reports/decision/gaps.csv`
- `reports/decision/summary.md`
- `reports/decision/figures/policy_comparison.png`
- `reports/decision/figures/occupancy_<policy>.png`
- `reports/decision/figures/urgent_protection_heatmap.png`
- `reports/decision/figures/divert_gap_heatmap.png`
- `reports/decision/figures/admit_curves.png`

The default recommendation objective is urgent-patient protection:

1. Maximize urgent/emergency admit rate.
2. Break ties by lower clinical cost per arrival.
3. Break remaining ties by lower mean boarding hours.
4. Break remaining ties by higher ICU utilization.

For a fast smoke test, reduce `--n-runs` to 2 and use a smaller grid such as
`--bed-grid 6 --arrival-rate-grid 1`.

The `reports/` directory contains generated outputs and is ignored by Git.

### What Each Figure Is For

- `policy_comparison.png`: compares admit, board, divert, transfer, urgent
  admit, and elective admit rates in the main scenario.
- `occupancy_<policy>.png`: shows mean ICU utilization and the 5-95% band
  across Monte-Carlo runs, useful for spotting chronic full-bed pressure or
  excessive idle capacity.
- `urgent_protection_heatmap.png`: shows where the recommended reservation
  policy improves urgent/emergency admit rate relative to FCFS.
- `divert_gap_heatmap.png`: shows whether the recommended policy also reduces
  diversion relative to FCFS.
- `admit_curves.png`: shows how urgent admit rates change as arrival pressure
  rises, faceted by ICU bed count.

### Main Demo Pipeline

`run-demo` is a smaller one-scenario pipeline for quick policy comparison:

```bash
icu-scheduler run-demo \
    --n-beds 2 \
    --reserves 1 \
    --compress-timeline \
    --forecast-max-board 2 \
    --forecast-max-board-wait 24 \
    --max-transfer-remaining-hours 24 \
    --n-runs 20 \
    --n-workers 1 \
    --output reports/final
```

The command writes:

- `reports/final/metrics.csv`
- `reports/final/figures/comparison.png`
- `reports/final/figures/occupancy_fcfs.png`
- `reports/final/figures/occupancy_threshold_r1.png`
- `reports/final/figures/occupancy_adaptive_threshold.png`
- `reports/final/figures/occupancy_forecast_ilp.png`

## 5. Current Result Snapshot

The latest checked smoke test used:

```bash
icu-scheduler run-demo \
    --n-beds 2 \
    --reserves 1 \
    --compress-timeline \
    --forecast-max-board 2 \
    --forecast-max-board-wait 24 \
    --max-transfer-remaining-hours 24 \
    --n-runs 20 \
    --n-workers 1 \
    --output reports/stepdown_compare_20
```

Result summary:

| Policy | Urgent admit rate | Board rate | Mean boarding hours | Transfer rate | Clinical cost / arrival |
|---|---:|---:|---:|---:|---:|
| FCFS | 59.16% | 39.74% | 108.17 | 0.00% | 22.563 |
| Threshold r=1 | 62.96% | 31.33% | 107.85 | 0.00% | 21.206 |
| Forecast ILP | 74.83% | 4.29% | 13.62 | 0.41% | 19.503 |

Interpretation:

- FCFS boards many patients and produces very long boarding waits.
- Threshold improves urgent protection by reserving capacity, but still boards
  many patients.
- Forecast ILP has the best urgent admit rate and the lowest clinical cost in
  this stress setting.
- Step-down transfer is rare by design. It only occurs when an urgent or
  emergency patient arrives, the ICU is full, and a low-acuity current ICU
  patient is close enough to discharge.

## 6. Allocation Policies

### FCFS

The first-come-first-served policy admits any arrival when a bed is open. If no
bed is open, it boards patients up to a finite queue size and then diverts.

### Threshold Reservation

The threshold policy reserves a fixed number of ICU beds for urgent/emergency
patients. Elective or observation patients are diverted once the number of free
beds is at or below the reserve level.

### Adaptive Threshold Reservation

`AdaptiveThresholdPolicy` changes the reserve level based on current ICU
congestion: no reserved beds below 70% occupancy, one reserved bed between 70%
and 90% occupancy, and two reserved beds at or above 90% occupancy. Urgent and
emergency arrivals still admit whenever any bed is free; the adaptive reserve
mainly changes elective admission decisions under load.

### Forecast-Driven Rolling ILP

The main optimized policy is `forecast_ilp`. At each patient arrival, it:

1. Observes current occupancy, current boarding queue, and scheduled discharges.
2. Uses a historical arrival/LOS forecaster to create a 72-hour demand scenario.
3. Solves a small integer linear program over the current arrival and predicted
   future arrivals.
4. Executes only the current decision.
5. Re-optimizes when the next patient arrives.

The decision variable is:

```text
x_i = 1 if patient i is admitted, 0 otherwise
```

For every time interval in the rolling horizon, the ILP enforces:

```text
sum(active_i_in_interval * x_i) <= ICU capacity
```

The objective maximizes weighted clinical benefit. Urgent and emergency
patients receive higher benefit than elective or observation patients.

### Finite Boarding

The simulator models boarding as a finite queue, not as unlimited hidden
capacity. A boarded patient waits until an ICU discharge actually occurs.
`forecast_ilp` only boards a patient when a bed is expected to open within
`--forecast-max-board-wait` hours.

### Conservative Step-Down Transfer

When an urgent or emergency patient arrives and the ICU is full, `forecast_ilp`
may recommend transferring a current low-acuity patient to step-down or ward
care. This is intentionally conservative. A transfer is allowed only when:

- The new arrival is `urgent` or `emergency`.
- The ICU is full.
- The current ICU patient is `elective` or `observation`.
- The current ICU patient has remaining predicted ICU LOS no larger than
  `--max-transfer-remaining-hours`.

Transfers are reported separately as `transfer_rate` and are penalized in the
clinical cost function.

## 7. Evaluation Metric

The main evaluation metric is a weighted clinical and operational cost:

```text
cost =
100 * urgent_diverted
 + 30 * urgent_boarded
 + 20 * elective_diverted
 + 10 * elective_boarded
 + 0.25 * boarding_hours
 +  8 * transferred
 +  1 * idle_bed_step
```

This makes urgent diversion the worst event, followed by urgent boarding,
elective diversion, elective boarding, transfer, and idle capacity. The same
priority structure is used by the forecast ILP admission benefits, so the
optimization objective and evaluation metric are aligned.

## 8. Command Reference

The package installs a command-line program named `icu-scheduler`. It has
built-in help at both the top level and subcommand level:

```bash
icu-scheduler --help
icu-scheduler decision-report --help
icu-scheduler run-demo --help
icu-scheduler ingest-mimic --help
icu-scheduler validate-db --help
icu-scheduler optimize --help
```

Top-level commands:

```text
ingest-mimic  Ingest MIMIC-IV CSVs into a SQLite cache without running simulation.
validate-db   Check that a MIMIC-IV SQLite file has the expected tables/columns.
decision-report  Generate the hospital-facing recommendation report.
run-demo      Run the end-to-end simulation and policy-comparison pipeline.
optimize      Solve the offline ILP bed-allocation benchmark.
```

### Which Command Should I Run?

- Final project result: use `icu-scheduler decision-report`.
- Quick smoke test: use `icu-scheduler run-demo --n-runs 2 --n-workers 1`.
- Offline theoretical best-with-hindsight benchmark: use `icu-scheduler optimize`.
- Data check before running simulations: use `icu-scheduler validate-db`.
- Convert raw MIMIC CSVs to SQLite cache: use `icu-scheduler ingest-mimic`.

### Test and Coverage

Run the full unit-test suite with coverage:

```bash
pytest --cov=icu_scheduler --cov-report=term-missing
```

Expected local result:

```text
137 passed
90% total coverage
```

### Decision Report

Run the hospital-facing report. This auto-discovers a MIMIC-shaped folder under
`data/`, ingests it to a cached SQLite database if needed, runs the main
scenario with FCFS, threshold reservation, adaptive threshold reservation, and
forecast ILP, then runs a faster bed-count by arrival-pressure sweep for
heatmaps and recommendations.

```bash
icu-scheduler decision-report \
    --n-beds 6 \
    --reserves 0,1,2,3,4,5,6 \
    --arrival-rate-grid 1,1.5,2,3,4 \
    --bed-grid 3,4,5,6,8 \
    --compress-timeline \
    --main-arrival-rate 1 \
    --forecast-max-board 2 \
    --forecast-max-board-wait 24 \
    --max-transfer-remaining-hours 24 \
    --n-runs 20 \
    --n-workers 1 \
    --output reports/decision
```

Important `decision-report` options:

- `--n-beds`: main-scenario ICU bed capacity.
- `--reserves`: comma-separated reservation values for `ThresholdPolicy`, such
  as `0,1,2,3`.
- `--bed-grid`: ICU capacities used for heatmaps and recommendations.
- `--arrival-rate-grid`: arrival-pressure multipliers used for heatmaps and
  recommendations.
- `--main-arrival-rate`: arrival-pressure multiplier for the main comparison
  that includes `forecast_ilp`.
- `--compress-timeline`: required for meaningful real MIMIC-IV Demo runs,
  because deidentified timestamps otherwise spread arrivals across decades.
- `--n-runs`: Monte-Carlo replications. Use small values for smoke tests and
  larger values for final figures.
- `--n-workers`: parallel simulation workers.
- `--output`: destination folder for `metrics.csv` and `figures/*.png`.

Output files:

```text
reports/decision/metrics.csv
reports/decision/scenario_metrics.csv
reports/decision/recommendations.csv
reports/decision/gaps.csv
reports/decision/summary.md
reports/decision/figures/policy_comparison.png
reports/decision/figures/occupancy_fcfs.png
reports/decision/figures/occupancy_threshold_r1.png
reports/decision/figures/occupancy_adaptive_threshold.png
reports/decision/figures/occupancy_forecast_ilp.png
reports/decision/figures/urgent_protection_heatmap.png
reports/decision/figures/divert_gap_heatmap.png
reports/decision/figures/admit_curves.png
```

If `data/` is missing or does not contain MIMIC-shaped files, this command
raises an error. Use `--mimic-dir /path/to/mimic` or `--db /path/to/cache.sqlite`
when the data is stored somewhere else.

### Main Demo Pipeline

Use `run-demo` for a smaller one-scenario smoke test. It now compares FCFS,
fixed threshold reservation, adaptive threshold reservation, and forecast ILP.

```bash
icu-scheduler run-demo \
    --n-beds 2 \
    --reserves 1 \
    --compress-timeline \
    --forecast-max-board 2 \
    --forecast-max-board-wait 24 \
    --max-transfer-remaining-hours 24 \
    --n-runs 20 \
    --n-workers 1 \
    --output reports/final
```

### Reserve Sweep

To compare many fixed reserve levels for a chosen bed count:

```bash
icu-scheduler run-demo \
    --n-beds 6 \
    --reserves 0,1,2,3,4,5,6 \
    --compress-timeline \
    --n-runs 50 \
    --n-workers 4 \
    --output reports/reserve_sweep
```

Then inspect the ranking by utilization:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("reports/reserve_sweep/metrics.csv")
cols = [
    "policy",
    "mean_utilization",
    "p95_utilization",
    "urgent_admit_rate",
    "elective_admit_rate",
    "clinical_cost_per_arrival",
]
print(df[cols].sort_values("mean_utilization", ascending=False).to_string(index=False))
PY
```

### Stress Sweep Figures

Run the bed-count by arrival-rate stress sweep used for the heatmaps:

```bash
python scripts/stress_sweep.py
```

Outputs:

```text
reports/stress_sweep/stress_grid.csv
reports/stress_sweep/gaps.csv
reports/stress_sweep/figures/divert_gap.png
reports/stress_sweep/figures/urgent_protection.png
reports/stress_sweep/figures/admit_curves.png
```

The stress-sweep script is intentionally a plain Python script rather than a
Click subcommand. To change the sweep grid, edit the constants at the top of
`scripts/stress_sweep.py`: `BEDS_GRID`, `RATES_GRID`, `RESERVES`, `N_RUNS`, and
`N_WORKERS`.

### MIMIC Ingestion and Validation

Validate a MIMIC SQLite database:

```bash
icu-scheduler validate-db data/mimic-iv-clinical-database-demo-2.2/.icu_scheduler_cache.sqlite
```

Prebuild the SQLite cache:

```bash
icu-scheduler ingest-mimic data/mimic-iv-clinical-database-demo-2.2
```

### Offline ILP Benchmark

Run the offline ILP optimizer. This is a best-possible-with-hindsight benchmark,
not the online rolling policy used by `run-demo`.

```bash
icu-scheduler optimize \
    --n-beds 2 \
    --urgent-benefit 100 \
    --elective-benefit 20 \
    --compress-timeline \
    --output reports/optimization
```

Output files:

```text
reports/optimization/decisions.csv
reports/optimization/summary.csv
```

### Documentation Build

Build local HTML documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

## 9. Module-Level Example

```python
from icu_scheduler.data_loader import load_joined_cohort
from icu_scheduler.scheduler import ThresholdPolicy
from icu_scheduler.simulator import MonteCarloSimulator
from icu_scheduler.stream import ArrivalStream
from icu_scheduler.evaluator import compute_kpis

cohort = load_joined_cohort("data/sample/mimic_demo.sqlite")
cohort = cohort.assign(los_hours=cohort["los"] * 24.0)
stream = ArrivalStream.from_dataframe(cohort)
policy = ThresholdPolicy(n_beds=6, reserve=1, max_board=5)

sim = MonteCarloSimulator(policy, stream, n_runs=100, n_workers=4)
results = sim.run()
print(compute_kpis(results))
```

## 10. Project Structure

```text
icu_scheduler_project/
  pyproject.toml
  README.md
  Makefile
  src/icu_scheduler/
    cli.py
    data_loader.py
    decision_support.py
    evaluator.py
    forecasting.py
    mapreduce.py
    mimic_ingest.py
    optimization.py
    plotting.py
    scheduler.py
    simulator.py
    stream.py
  tests/
    test_cli.py
    test_data_loader.py
    test_decision_support.py
    test_evaluator.py
    test_forecasting.py
    test_mimic_ingest.py
    test_optimization.py
    test_plotting.py
    test_scheduler.py
    test_simulator.py
    test_stream.py
  scripts/
    build_sample_db.py
    run_real_mimic.py
    stress_sweep.py
  docs/
    conf.py
    index.rst
    api.rst
  reports/
    stepdown_compare_20/
    stress_sweep/
    optimization/
    # generated outputs, ignored by Git
```

## 11. Course Skills Used

| Course skill | Application in this project |
|---|---|
| Python packaging | `pyproject.toml`, `src/` layout, CLI entry point |
| SQL | MIMIC-IV ingestion and joins with SQLAlchemy |
| Unit testing | 137 passing tests with 90% coverage |
| Data structures | Priority queue for scheduled ICU discharges |
| Data streams | Arrival stream abstraction, reservoir sampling, count-min sketch |
| Multiprocessing | Parallel Monte-Carlo simulation runs |
| MapReduce | Care-unit hour aggregation |
| Optimization | Offline ILP benchmark and online rolling-horizon ILP |
| Documentation | README plus Sphinx documentation skeleton |

## 12. Limitations

- The main public experiment uses MIMIC-IV Demo v2.2, which is much smaller
  than the full MIMIC-IV database.
- The forecast model is intentionally simple and deployable: it estimates
  acuity-specific arrival rates and median LOS from historical data.
- Step-down transfer is a conservative operational approximation. A real
  deployment would require clinical eligibility rules, staffing constraints,
  and external validation.
- The objective weights are transparent and configurable, but they are not
  learned from patient outcomes.

## 13. License and Data Use

This project code is released under the MIT License. MIMIC-IV data is governed
by the PhysioNet data use agreement and should be handled according to
PhysioNet's terms.

## 14. Acknowledgements

- Johnson et al., MIMIC-IV, PhysioNet.
- ORIE 5270/6125 teaching staff.
