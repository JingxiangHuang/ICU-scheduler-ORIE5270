"""Tests for icu_scheduler.cli.

Uses ``click.testing.CliRunner`` to invoke the CLI in-process.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from icu_scheduler.cli import main


class TestValidateDb:
    def test_valid_db_succeeds(self, tiny_mimic_db, tmp_path):
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(main, ["validate-db", sqlite_path])
        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_missing_table_fails(self, tmp_path):
        # Build a DB that's missing the 'icustays' table.
        import pandas as pd
        from sqlalchemy import create_engine

        bad = tmp_path / "bad.sqlite"
        engine = create_engine(f"sqlite:///{bad}", future=True)
        with engine.begin() as conn:
            pd.DataFrame({"x": [1]}).to_sql("unrelated", conn, index=False)

        runner = CliRunner()
        result = runner.invoke(main, ["validate-db", str(bad)])
        assert result.exit_code != 0


class TestRunDemo:
    def test_run_demo_with_tiny_db(self, tiny_mimic_db, tmp_path):
        """Run the end-to-end pipeline on the fixture DB."""
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        output_dir = tmp_path / "reports"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "3",
                "--n-workers",
                "1",
                "--output",
                str(output_dir),
            ],
        )
        # Print output on failure for easier debugging.
        assert result.exit_code == 0, result.output
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "figures" / "comparison.png").exists()


class TestCliGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run-demo" in result.output
        assert "decision-report" in result.output
        assert "optimize" in result.output
        assert "validate-db" in result.output
        assert "ingest-mimic" in result.output


class TestOptimize:
    def test_optimize_with_tiny_db(self, tiny_mimic_db, tmp_path):
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        output_dir = tmp_path / "opt"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "optimize",
                "--db",
                sqlite_path,
                "--n-beds",
                "2",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Objective value" in result.output
        assert (output_dir / "decisions.csv").exists()
        assert (output_dir / "summary.csv").exists()

    def test_optimize_errors_without_default_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["optimize", "--output", "opt"])

        assert result.exit_code != 0
        assert "No ./data directory found" in result.output
        assert "--mimic-dir" in result.output
        assert "--db" in result.output


class TestRunDemoParameters:
    """The user-facing tuning knobs — --n-beds, --reserves, etc. — are the
    main knobs this project is meant to expose. Lock them down."""

    def test_n_beds_and_reserves_grid(self, tiny_mimic_db, tmp_path):
        """--n-beds and --reserves actually change the policy grid."""
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "3",
                "--n-beds",
                "8",
                "--reserves",
                "1,2,3",
                "--output",
                str(tmp_path / "reports"),
            ],
        )
        assert result.exit_code == 0, result.output
        # Grid should be fcfs + threshold_r1 + threshold_r2 + threshold_r3.
        assert "threshold_r1" in result.output
        assert "threshold_r2" in result.output
        assert "threshold_r3" in result.output
        assert "forecast_ilp" in result.output
        assert "n_beds=8" in result.output
        # Matching figure files should land on disk.
        figs = list((tmp_path / "reports" / "figures").glob("occupancy_*.png"))
        names = {f.stem for f in figs}
        assert {
            "occupancy_fcfs",
            "occupancy_threshold_r1",
            "occupancy_threshold_r2",
            "occupancy_threshold_r3",
            "occupancy_adaptive_threshold",
            "occupancy_forecast_ilp",
        } == names

    def test_bad_reserves_raises(self, tiny_mimic_db, tmp_path):
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--reserves",
                "1,not-an-int,3",
                "--output",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code != 0
        assert "reserves" in result.output.lower()

    def test_compress_timeline_flag(self, tiny_mimic_db, tmp_path):
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "2",
                "--compress-timeline",
                "--output",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Timeline compressed" in result.output

    def test_arrival_rate_multiplier_flag_logs(self, tiny_mimic_db, tmp_path):
        """--arrival-rate-multiplier != 1 logs a stress-test line."""
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "2",
                "--compress-timeline",
                "--arrival-rate-multiplier",
                "3.0",
                "--output",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Arrival-rate stress test" in result.output

    def test_arrival_rate_multiplier_default_is_silent(self, tiny_mimic_db, tmp_path):
        """multiplier=1.0 is a no-op: no stress-test log line."""
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "2",
                "--output",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Arrival-rate stress test" not in result.output

    def test_arrival_rate_multiplier_rejects_zero(self, tiny_mimic_db, tmp_path):
        """multiplier <= 0 should fail loudly — silent division by zero
        would otherwise be very confusing."""
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--db",
                sqlite_path,
                "--n-runs",
                "2",
                "--arrival-rate-multiplier",
                "0",
                "--output",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code != 0
        assert "must be > 0" in result.output or "arrival-rate-multiplier" in result.output.lower()


class TestAutoDiscover:
    """Users drop a MIMIC folder under ./data/ and run `icu-scheduler run-demo`
    with no flags — these tests pin down that behaviour."""

    @staticmethod
    def _write_tiny_mimic(root, subdir_name=None):
        """Populate a MIMIC-shaped CSV pair under ``root/<subdir_name?>/``."""
        import pandas as pd

        base = root if subdir_name is None else root / subdir_name
        (base / "hosp").mkdir(parents=True)
        (base / "icu").mkdir(parents=True)

        adm = pd.DataFrame(
            [
                {
                    "subject_id": i,
                    "hadm_id": i * 10,
                    "admittime": f"2180-01-{(i % 28) + 1:02d} 08:00:00",
                    "dischtime": f"2180-01-{(i % 28) + 2:02d} 08:00:00",
                    "admission_type": "EW EMER." if i % 2 else "ELECTIVE",
                    "admission_location": "ER",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                }
                for i in range(1, 6)
            ]
        )
        icu = pd.DataFrame(
            [
                {
                    "subject_id": i,
                    "hadm_id": i * 10,
                    "stay_id": i * 100,
                    "first_careunit": "MICU",
                    "last_careunit": "MICU",
                    "intime": f"2180-01-{(i % 28) + 1:02d} 09:00:00",
                    "outtime": f"2180-01-{(i % 28) + 2:02d} 09:00:00",
                    "los": 1.0,
                }
                for i in range(1, 6)
            ]
        )
        adm.to_csv(base / "hosp" / "admissions.csv.gz", index=False, compression="gzip")
        icu.to_csv(base / "icu" / "icustays.csv.gz", index=False, compression="gzip")
        return base

    def test_discovers_data_folder_without_flags(self, tmp_path, monkeypatch):
        """With CSVs in ./data/ the CLI should run with no --mimic-dir flag."""
        self._write_tiny_mimic(tmp_path / "data", subdir_name=None)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run-demo", "--n-runs", "3", "--output", "reports"],
        )
        assert result.exit_code == 0, result.output
        assert "Auto-discovered MIMIC data" in result.output
        assert (tmp_path / "reports" / "metrics.csv").exists()

    def test_discovers_nested_folder_under_data(self, tmp_path, monkeypatch):
        """Folder like data/mimic-iv-clinical-database-demo-2.2/ still works."""
        self._write_tiny_mimic(tmp_path / "data", subdir_name="mimic-iv-clinical-database-demo-2.2")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["run-demo", "--n-runs", "3", "--output", "reports"])
        assert result.exit_code == 0, result.output
        assert "Auto-discovered MIMIC data" in result.output

    def test_errors_when_data_directory_missing(self, tmp_path, monkeypatch):
        """No implicit fallback: missing ./data should tell users how to proceed."""
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["run-demo", "--n-runs", "2", "--output", "reports"])

        assert result.exit_code != 0
        assert "No ./data directory found" in result.output
        assert "--mimic-dir" in result.output
        assert "--db" in result.output

    def test_errors_when_data_empty(self, tmp_path, monkeypatch):
        """Empty data/ should fail loudly instead of using sample/synthetic data."""
        (tmp_path / "data").mkdir()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["run-demo", "--n-runs", "2", "--output", "reports"])
        assert result.exit_code != 0
        assert "No MIMIC-shaped dataset found" in result.output
        assert "--mimic-dir" in result.output
        assert "--db" in result.output


class TestDecisionReport:
    def test_decision_report_with_tiny_db(self, tiny_mimic_db, tmp_path):
        sqlite_path = tiny_mimic_db.replace("sqlite:///", "")
        output_dir = tmp_path / "decision"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "decision-report",
                "--db",
                sqlite_path,
                "--n-runs",
                "2",
                "--n-workers",
                "1",
                "--n-beds",
                "2",
                "--reserves",
                "0,1,2",
                "--bed-grid",
                "2",
                "--arrival-rate-grid",
                "1",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Decision report complete" in result.output
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "scenario_metrics.csv").exists()
        assert (output_dir / "recommendations.csv").exists()
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "figures" / "policy_comparison.png").exists()
        assert (output_dir / "figures" / "urgent_protection_heatmap.png").exists()
        assert (output_dir / "figures" / "divert_gap_heatmap.png").exists()
        assert (output_dir / "figures" / "admit_curves.png").exists()
        assert (output_dir / "figures" / "occupancy_fcfs.png").exists()
        assert (output_dir / "figures" / "occupancy_adaptive_threshold.png").exists()
        assert (output_dir / "figures" / "occupancy_forecast_ilp.png").exists()

    def test_decision_report_errors_without_default_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "decision-report",
                "--n-runs",
                "1",
                "--bed-grid",
                "2",
                "--arrival-rate-grid",
                "1",
                "--output",
                "decision",
            ],
        )

        assert result.exit_code != 0
        assert "No ./data directory found" in result.output
        assert "--mimic-dir" in result.output
        assert "--db" in result.output


class TestIngestMimic:
    def test_ingest_then_run_demo(self, tmp_path):
        """End-to-end: unzipped MIMIC folder -> auto-ingest -> run simulation."""
        import pandas as pd

        mimic = tmp_path / "mimic"
        (mimic / "hosp").mkdir(parents=True)
        (mimic / "icu").mkdir(parents=True)

        adm = pd.DataFrame(
            [
                {
                    "subject_id": i,
                    "hadm_id": i * 10,
                    "admittime": f"2180-01-{(i % 28) + 1:02d} 08:00:00",
                    "dischtime": f"2180-01-{(i % 28) + 2:02d} 08:00:00",
                    "admission_type": "EW EMER." if i % 2 else "ELECTIVE",
                    "admission_location": "ER",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                }
                for i in range(1, 11)
            ]
        )
        icu = pd.DataFrame(
            [
                {
                    "subject_id": i,
                    "hadm_id": i * 10,
                    "stay_id": i * 100,
                    "first_careunit": "MICU",
                    "last_careunit": "MICU",
                    "intime": f"2180-01-{(i % 28) + 1:02d} 09:00:00",
                    "outtime": f"2180-01-{(i % 28) + 2:02d} 09:00:00",
                    "los": 1.0,
                }
                for i in range(1, 11)
            ]
        )
        adm.to_csv(mimic / "hosp" / "admissions.csv.gz", index=False, compression="gzip")
        icu.to_csv(mimic / "icu" / "icustays.csv.gz", index=False, compression="gzip")

        output_dir = tmp_path / "reports"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run-demo",
                "--mimic-dir",
                str(mimic),
                "--n-runs",
                "3",
                "--n-workers",
                "1",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Ingesting MIMIC dir" in result.output
        assert (output_dir / "metrics.csv").exists()
        # Cache should be sitting in the MIMIC dir for next time.
        assert (mimic / ".icu_scheduler_cache.sqlite").exists()
