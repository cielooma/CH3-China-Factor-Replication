"""Reproducibility plumbing: fingerprints, manifests and the registry."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant_research.repro import (
    REGISTRY_COLUMNS,
    append_registry_row,
    build_manifest,
    canonical_json,
    fingerprint,
    fingerprint_file,
    git_commit,
    read_registry,
)


def registry_row(experiment_id: str, **overrides) -> dict[str, str]:
    row = {column: "" for column in REGISTRY_COLUMNS}
    row["experiment_id"] = experiment_id
    row.update({key: str(value) for key, value in overrides.items()})
    return row


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_across_key_order(self) -> None:
        first = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": "z"}}
        second = {"nested": {"x": "z", "y": [1, 2]}, "a": 1, "b": 2}
        self.assertEqual(fingerprint(first), fingerprint(second))
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_fingerprint_changes_when_a_value_changes(self) -> None:
        self.assertNotEqual(fingerprint({"a": 1}), fingerprint({"a": 2}))

    def test_fingerprint_handles_dates_and_tuples(self) -> None:
        import datetime

        payload = {"breakpoints": (0.3, 0.7), "date": datetime.date(2016, 12, 31)}
        self.assertEqual(len(fingerprint(payload)), 16)
        json.loads(canonical_json(payload))

    def test_file_fingerprint_tracks_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            first = fingerprint_file(path)
            path.write_text("a,b\n1,3\n", encoding="utf-8")
            self.assertNotEqual(first, fingerprint_file(path))

    def test_missing_file_is_a_loud_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            fingerprint_file("/nonexistent/path/for/a/test.csv")


class ManifestTests(unittest.TestCase):
    def test_manifest_records_config_inputs_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            market = Path(directory) / "market.csv"
            market.write_text("security_id,observation_date\n000001.SZ,2000-01-31\n", encoding="utf-8")
            config = {"sample_start": "2000-01-31", "exclude_smallest_fraction": 0.3}
            manifest = build_manifest(
                experiment_id="EXP-TEST-001",
                config=config,
                inputs={"market": market},
                panel_fingerprint="deadbeefdeadbeef",
                root=directory,
                extra={"note": "unit test"},
            )

        self.assertEqual(manifest["experiment_id"], "EXP-TEST-001")
        self.assertEqual(manifest["config_fingerprint"], fingerprint(config))
        self.assertEqual(manifest["panel_fingerprint"], "deadbeefdeadbeef")
        self.assertTrue(manifest["inputs"]["market"]["exists"])
        self.assertEqual(len(manifest["inputs"]["market"]["fingerprint"]), 16)
        self.assertIn("python", manifest["environment"])
        self.assertEqual(manifest["code_commit"], "uncommitted")
        self.assertEqual(manifest["manifest_fingerprint"], manifest["manifest_fingerprint"])

    def test_manifest_fingerprint_ignores_the_timestamp(self) -> None:
        config = {"a": 1}
        first = build_manifest(experiment_id="EXP", config=config, root=".")
        second = build_manifest(experiment_id="EXP", config=config, root=".")
        self.assertEqual(first["manifest_fingerprint"], second["manifest_fingerprint"])

    def test_git_commit_reports_a_dirty_tree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            self.skipTest("not a git checkout")
        commit = git_commit(root)
        self.assertNotEqual(commit, "")


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.registry = Path(self.directory.name) / "experiment_registry.csv"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_first_write_appends_and_second_write_updates(self) -> None:
        self.assertEqual(
            append_registry_row(self.registry, registry_row("EXP-1", status="registered")),
            "appended",
        )
        self.assertEqual(
            append_registry_row(self.registry, registry_row("EXP-1", status="completed")),
            "updated",
        )
        rows = read_registry(self.registry)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "completed")

    def test_a_second_experiment_does_not_disturb_the_first(self) -> None:
        append_registry_row(self.registry, registry_row("EXP-1"))
        append_registry_row(self.registry, registry_row("EXP-2"))
        self.assertEqual([row["experiment_id"] for row in read_registry(self.registry)], ["EXP-1", "EXP-2"])

    def test_header_order_is_preserved(self) -> None:
        append_registry_row(self.registry, registry_row("EXP-1"))
        with self.registry.open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, list(REGISTRY_COLUMNS))

    def test_missing_and_unknown_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown columns"):
            append_registry_row(self.registry, {**registry_row("EXP-1"), "typo_column": "x"})
        incomplete = registry_row("EXP-1")
        incomplete.pop("notes")
        with self.assertRaisesRegex(KeyError, "missing required columns"):
            append_registry_row(self.registry, incomplete)

    def test_existing_registry_is_read_back(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = root / "00_admin" / "experiment_registry.csv"
        if not registry.exists():
            self.skipTest("no registry in the workspace")
        rows = read_registry(registry)
        self.assertTrue(rows)
        self.assertEqual(set(rows[0]), set(REGISTRY_COLUMNS))


if __name__ == "__main__":
    unittest.main()
