"""End-to-end tests for the CH-3 pipeline entry point.

These run the script as a subprocess, because the behaviour under test is the
script's: its exit code, its files, and whether the persisted leakage report
tells the truth about a run that was rejected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ch3_reproduction.py"


def build_fixture(directory: Path) -> dict[str, Path]:
    sys.path.insert(0, str(ROOT / "src"))
    from quant_research.synthetic import SyntheticPanelSpec, simulate_ch3_panel

    bundle = simulate_ch3_panel(
        SyntheticPanelSpec(n_stocks=150, start="2005-01", end="2008-12")
    )
    paths = {
        "market": directory / "market.csv",
        "fundamentals": directory / "fundamentals.csv",
        "author": directory / "author.csv",
        "leaky": directory / "market_leaky.csv",
    }
    bundle.market.to_csv(paths["market"], index=False)
    bundle.fundamentals.to_csv(paths["fundamentals"], index=False)
    bundle.author_factors.to_csv(paths["author"])
    bundle.market_leaky.to_csv(paths["leaky"], index=False)
    return paths


class PipelineScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls._temp.name)
        cls.paths = build_fixture(cls.directory)
        cls.env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=self.env,
            cwd=str(ROOT),
            check=False,
        )

    def test_a_clean_panel_runs_and_registers(self) -> None:
        output = self.directory / "clean_out"
        registry = self.directory / "clean_registry.csv"
        result = self.run_script(
            "--market", str(self.paths["market"]),
            "--fundamentals", str(self.paths["fundamentals"]),
            "--author", str(self.paths["author"]),
            "--output", str(output),
            "--registry", str(registry),
            "--no-full-logs",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        for name in (
            "factor_returns.csv",
            "diagnostics.csv",
            "breakpoints.csv",
            "alignment.csv",
            "leakage_report.json",
            "run_manifest.json",
            "summary.json",
        ):
            self.assertTrue((output / name).exists(), f"{name} was not written")

        leakage = json.loads((output / "leakage_report.json").read_text())
        self.assertTrue(leakage["passed"])

        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(summary["alignment"]), 5)
        self.assertFalse(summary["leakage_overridden"])

        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["experiment_id"], "EXP-CH3-001")
        self.assertEqual(len(manifest["config_fingerprint"]), 16)
        self.assertTrue(manifest["inputs"]["market"]["exists"])

        rows = registry.read_text().strip().splitlines()
        self.assertEqual(len(rows), 2)  # header + one experiment
        self.assertIn("EXP-CH3-001", rows[1])

    def test_a_leaky_panel_is_rejected_with_a_failing_report(self) -> None:
        output = self.directory / "leaky_out"
        result = self.run_script(
            "--market", str(self.paths["leaky"]),
            "--fundamentals", str(self.paths["fundamentals"]),
            "--output", str(output),
            "--no-registry",
        )
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertFalse((output / "factor_returns.csv").exists())

        report = json.loads((output / "leakage_report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertGreater(report["n_errors"], 0)
        self.assertIn("availability", [finding["code"] for finding in report["findings"]])

    def test_required_alignment_cannot_pass_without_a_reference(self) -> None:
        result = self.run_script(
            "--market", str(self.paths["market"]),
            "--fundamentals", str(self.paths["fundamentals"]),
            "--author", str(self.directory / "missing_reference.csv"),
            "--output", str(self.directory / "no_reference"),
            "--require-alignment", "--no-registry", "--no-full-logs",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_the_override_is_recorded_rather_than_silent(self) -> None:
        output = self.directory / "override_out"
        result = self.run_script(
            "--market", str(self.paths["leaky"]),
            "--fundamentals", str(self.paths["fundamentals"]),
            "--author", str(self.paths["author"]),
            "--output", str(output),
            "--no-registry",
            "--no-full-logs",
            "--allow-leakage",
        )
        # Non-zero on purpose: the analysis completed, but a caller must not be
        # able to mistake the override for a clean run.
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["leakage_overridden"])
        report = json.loads((output / "leakage_report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertIn("availability", [f["code"] for f in report["findings"]])

    def test_a_rejected_panel_never_produces_a_passing_report(self) -> None:
        """A contract rejection must appear in the persisted report, not only stderr."""

        broken = self.directory / "duplicated_market.csv"
        import pandas as pd

        market = pd.read_csv(self.paths["market"])
        duplicated = pd.concat([market, market.head(1)], ignore_index=True)
        duplicated.to_csv(broken, index=False)

        output = self.directory / "duplicate_out"
        result = self.run_script(
            "--market", str(broken),
            "--fundamentals", str(self.paths["fundamentals"]),
            "--output", str(output),
            "--no-registry",
        )
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        report = json.loads((output / "leakage_report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertIn("panel_contract", [finding["code"] for finding in report["findings"]])


if __name__ == "__main__":
    unittest.main()
