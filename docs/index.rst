ICU Scheduler — documentation
==============================

A reproducible Python toolkit for optimizing ICU bed allocation using the
MIMIC-IV clinical dataset. The project is packaged with a ``src/`` layout,
a command-line interface, unit tests, and Sphinx API documentation.

Purpose
-------

ICU beds are scarce, and first-come-first-served admission can leave urgent
patients waiting when beds have already been filled by lower-acuity arrivals.
ICU Scheduler compares FCFS, fixed threshold reservation, adaptive threshold
reservation, and forecast-driven rolling-horizon optimization policies.

Dataset
-------

The main experiments use the MIMIC-IV Clinical Database Demo v2.2 from
PhysioNet. Raw MIMIC files are not committed to the repository; download the
demo dataset separately and place it under ``data/`` before running the
real-data scripts. If ``data/`` is missing or does not contain MIMIC-shaped
files, the CLI raises an explicit error instead of falling back to sample or
synthetic data.

Install and Test
----------------

From the project root::

   pip install -e ".[dev]"
   pytest --cov=icu_scheduler --cov-report=term-missing

The current local verification is 137 passing tests with 90% total coverage.

Run the Decision Report
-----------------------

After placing MIMIC-IV Demo v2.2 under ``data/``, run::

   icu-scheduler decision-report --n-beds 6 --reserves 0,1,2,3,4,5,6

This writes ``metrics.csv``, ``recommendations.csv``, ``summary.md``,
heatmaps, policy comparison charts, and occupancy curves under
``reports/decision/``. Use ``icu-scheduler run-demo`` for a faster
single-scenario smoke test.

.. toctree::
   :maxdepth: 2
   :caption: API Navigation:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
