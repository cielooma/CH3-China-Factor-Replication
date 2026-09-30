"""Run the CH-3 reproduction end to end.

    python scripts/run_ch3_reproduction.py                       # synthetic smoke test
    python scripts/run_ch3_reproduction.py --config configs/ch3_reference_v1.json \
        --market private_data/ch3_monthly_market.csv \
        --fundamentals private_data/ch3_fundamentals.csv \
        --author author_data/ch3_author_factors_monthly.csv

The sequence is fixed and deliberate:

1. load the point-in-time panel and refuse it if it breaks the time contract;
2. build the factors from the frozen config, with no full-sample statistic;
3. check the factor series for internal consistency and traceability;
4. align against the reference series and report the gaps;
5. write every output next to a manifest, and register the experiment.

A correlation below target is a *result*, not a crash: the run still completes
and writes its evidence.  ``--require-alignment`` turns the target into an exit
code, which is what the CI-style ``make ch3-smoke`` target uses.

Exit codes, so a caller can never mistake a flagged run for a clean one:

===== ==========================================================================
0     clean: the point-in-time gate passed and no quality check complained
1     completed but flagged: a quality check failed, or ``--allow-leakage`` let a
      run finish whose leakage report still says FAIL
2     unusable inputs: missing extracts, or a data version that does not match
      the frozen config
3     rejected by the point-in-time gate: no factor number was produced at all
===== ==========================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_research.factors import Ch3Spec, build_ch3_factors, validate_ch3_result  # noqa: E402
from quant_research.contracts import PointInTimeValidationError  # noqa: E402
from quant_research.pit import (  # noqa: E402
    ERROR,
    CSVDataProvider,
    LeakageError,
    LeakageFinding,
    PointInTimePanel,
    check_availability,
    check_chronological_split,
    check_execution_timing,
    check_factor_series_completeness,
    check_scaling_window,
    describe,
    describe_panel,
)
from quant_research.repro import (  # noqa: E402
    append_registry_row,
    build_manifest,
    write_manifest,
)

FACTORS = ("MKT", "SMB", "VMG", "HML_BM", "SMB_FF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ch3_reproduction_v1.json")
    parser.add_argument("--market", type=Path, default=None)
    parser.add_argument("--fundamentals", type=Path, default=None)
    parser.add_argument("--author", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=ROOT / "00_admin" / "experiment_registry.csv")
    parser.add_argument("--label", default=None)
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--no-full-logs", action="store_true")
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="continue despite a leakage error; the override is recorded in the manifest",
    )
    parser.add_argument(
        "--require-alignment",
        action="store_true",
        help="exit non-zero when a factor misses the correlation target",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "ch3" not in config or "run" not in config:
        raise SystemExit(f"{path} must contain both a 'ch3' recipe and a 'run' section")
    return config


def align_to_reference(
    factor_returns: pd.DataFrame,
    reference: pd.DataFrame,
    target_corr: float,
) -> pd.DataFrame:
    """Compare each reproduced factor with the published (or latent) series."""

    rows: list[dict[str, Any]] = []
    for factor in FACTORS:
        if factor not in factor_returns.columns:
            continue
        if factor not in reference.columns:
            rows.append({"factor": factor, "status": "absent from reference series"})
            continue
        reproduced = factor_returns[factor]
        published = reference[factor]
        joined = pd.concat(
            [reproduced.rename("reproduced"), published.rename("reference")], axis=1
        ).dropna()
        if len(joined) < 12:
            rows.append({"factor": factor, "status": "fewer than 12 overlapping months"})
            continue
        correlation = float(joined["reproduced"].corr(joined["reference"]))
        rows.append(
            {
                "factor": factor,
                "status": "meets target" if correlation >= target_corr else "below target",
                "n_months": int(len(joined)),
                "correlation": round(correlation, 4),
                "target_corr": target_corr,
                "rmse_pct": float(((joined["reproduced"] - joined["reference"]) ** 2).mean() ** 0.5 * 100),
                "max_abs_gap_pct": float((joined["reproduced"] - joined["reference"]).abs().max() * 100),
                "reproduced_mean_pct": round(float(joined["reproduced"].mean() * 100.0), 4),
                "reference_mean_pct": round(float(joined["reference"].mean() * 100.0), 4),
                "mean_gap_pct": round(
                    float((joined["reproduced"].mean() - joined["reference"].mean()) * 100.0), 4
                ),
                "reproduced_std_pct": round(float(joined["reproduced"].std(ddof=1) * 100.0), 4),
                "reference_std_pct": round(float(joined["reference"].std(ddof=1) * 100.0), 4),
                "std_gap_pct": round(
                    float((joined["reproduced"].std(ddof=1) - joined["reference"].std(ddof=1)) * 100.0),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows)


def load_panel(provider: CSVDataProvider, as_of: str) -> tuple[PointInTimePanel, Exception | None]:
    """Load strictly, but keep the broken panel so the failure can be explained.

    The strict constructor refuses to build a panel that breaks the time
    contract, which is the right default.  A rejection is much more useful as a
    written list of offending rows than as a stack trace, so the runner
    re-reads the same files unvalidated and reports what it finds.
    """

    try:
        return provider.load(as_of=as_of), None
    except PointInTimeValidationError as error:
        soft = PointInTimePanel.from_csv(
            provider.market_path, provider.fundamentals_path, validate=False
        )
        return soft, error


def _check_factor_timing(spec: Ch3Spec):
    """The factor timing convention, checked rather than asserted in prose.

    A factor observation is formed at the end of month ``t`` and earns its
    return over month ``t+1``.  That is the claim the whole exercise rests on,
    so it goes through the same execution-timing check any strategy would: a
    signal observed at a close may not be filled at that same close.
    """

    return check_execution_timing(
        pd.Timestamp(spec.sample_start),
        pd.Timestamp(spec.sample_start) + pd.DateOffset(months=1),
        label="CH-3 factor formation",
    )


def read_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.set_index(date_column).sort_index()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    spec = Ch3Spec.from_dict(config["ch3"])
    run_meta = config["run"]

    market_path = args.market or ROOT / run_meta.get(
        "market_path", "synthetic_data/ch3_monthly_market.csv"
    )
    fundamentals_path = args.fundamentals or ROOT / run_meta.get(
        "fundamentals_path", "synthetic_data/ch3_fundamentals.csv"
    )
    author_path = args.author or ROOT / run_meta.get(
        "author_path", "synthetic_data/ch3_author_factors.csv"
    )
    output_dir = args.output or ROOT / run_meta.get(
        "output_path", "private_outputs/ch3_reproduction_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or run_meta.get("label", spec.strategy_name)
    target_corr = float(run_meta.get("alignment_target_corr", 0.90))

    # ---------------------------------------------------------------- 1. load
    provider = CSVDataProvider(market_path, fundamentals_path)
    if not market_path.exists() or not fundamentals_path.exists():
        print(f"missing input data: {market_path} / {fundamentals_path}", file=sys.stderr)
        print("For the synthetic smoke test run `make ch3-data` first.", file=sys.stderr)
        return 2

    panel, load_error = load_panel(provider, spec.sample_end)

    # ------------------------------------------------------- 2. leakage gate
    # The checks run on the *unvalidated* panel, so the report is evidence
    # rather than a copy of the constructor's verdict.  A contract rejection is
    # folded in as its own finding, which is what stops a rejected run from
    # leaving behind a report that says PASS.
    leakage = check_availability(panel)
    if load_error is not None:
        leakage.add(
            LeakageFinding(
                code="panel_contract",
                severity=ERROR,
                message=f"the panel was rejected before any factor was built: {load_error}",
                evidence={"rejection": str(load_error)},
            )
        )
    leakage.extend(check_chronological_split(run_meta.get("periods", {}), require_convention=False))
    leakage.extend(check_scaling_window(run_meta.get("features", [])))
    leakage.extend(_check_factor_timing(spec))
    print(describe(leakage))

    if load_error is not None and not args.allow_leakage:
        (output_dir / "leakage_report.json").write_text(
            json.dumps(leakage.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"\nreport written to {output_dir / 'leakage_report.json'}\n"
            "No factor number is produced from a panel that cannot be dated. Fix the "
            "extract, or pass --allow-leakage to record an explicit override.",
            file=sys.stderr,
        )
        return 3

    leakage.raise_if_errors("CH-3 reproduction", allow_override=args.allow_leakage)

    if panel.data_version != spec.data_version:
        print(
            f"data version mismatch: the panel is {panel.data_version!r} but the frozen "
            f"config expects {spec.data_version!r}. Refusing to run: a result must name "
            "the extract it came from.",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------- 3. factors
    result = build_ch3_factors(panel, spec)
    for factor in result.factor_returns.columns:
        if factor in FACTORS:
            leakage.extend(
                check_factor_series_completeness(
                    result.factor_returns[factor], label=f"factor {factor}"
                )
            )
    # The report grew after the gate, so gate again: a completeness failure is
    # exactly the kind of late finding that must not be left unenforced.
    leakage.raise_if_errors("CH-3 factor series", allow_override=args.allow_leakage)

    problems = validate_ch3_result(result)
    if problems:
        print("\nfactor quality checks FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        (output_dir / "quality_report.json").write_text(json.dumps({"passed": False, "problems": problems}, indent=2))
        result.assignments.to_csv(output_dir / "rejected_assignments.csv", index=False)
        return 1

    # ------------------------------------------------------------ 4. alignment
    alignment = pd.DataFrame()
    if author_path.exists():
        reference = read_reference(author_path)
        alignment = align_to_reference(result.factor_returns, reference, target_corr)
    else:
        print(f"\nno reference series at {author_path}; skipping alignment")

    # -------------------------------------------------------------- 5. output
    result.factor_returns.to_csv(output_dir / "factor_returns.csv", float_format="%.8f")
    result.portfolio_returns.to_csv(output_dir / "portfolio_returns.csv", index=False, float_format="%.8f")

    diagnostics = result.diagnostics.drop(columns=["breakpoints", "portfolio_counts"])
    diagnostics.to_csv(output_dir / "diagnostics.csv", float_format="%.8f")
    pd.DataFrame(
        [{"date": date, **values} for date, values in result.diagnostics["breakpoints"].items()]
    ).to_csv(output_dir / "breakpoints.csv", index=False, float_format="%.8f")
    pd.DataFrame(
        [
            {"date": date, **counts}
            for date, counts in result.diagnostics["portfolio_counts"].items()
        ]
    ).to_csv(output_dir / "portfolio_counts.csv", index=False)

    exclusion = (
        result.universe_log.groupby(["formation_month", "exclusion_reason"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    exclusion.to_csv(output_dir / "exclusion_summary.csv")
    if not args.no_full_logs:
        result.universe_log.to_csv(output_dir / "universe_log.csv", index=False, float_format="%.8f")
        result.assignments.to_csv(output_dir / "portfolio_assignments.csv", index=False, float_format="%.8f")

    if not alignment.empty:
        alignment.to_csv(output_dir / "alignment.csv", index=False)
    (output_dir / "leakage_report.json").write_text(
        json.dumps(leakage.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "label": label,
        "config_path": str(args.config),
        "panel": describe_panel(
            panel, market_path=market_path, fundamentals_path=fundamentals_path
        ),
        "factors": result.summary()["factors"],
        "correlations": result.summary()["correlations"],
        "notes": result.notes,
        "leakage_passed": leakage.passed,
        "leakage_overridden": bool(args.allow_leakage and not leakage.passed),
        "alignment_target_corr": target_corr,
        "alignment": alignment.to_dict("records") if not alignment.empty else [],
        "quality_problems": problems,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = build_manifest(
        experiment_id=spec.experiment_id,
        config=config,
        inputs={
            "market": market_path,
            "fundamentals": fundamentals_path,
            **({"author_factors": author_path} if author_path.exists() else {}),
        },
        panel_fingerprint=panel.fingerprint(),
        root=ROOT,
        label=label,
        extra={
            "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), ROOT / "src/quant_research/factors/ch3.py", ROOT / "src/quant_research/pit/panel.py"]},
            "config_fingerprint_ch3": spec.fingerprint(),
            "n_factor_months": int(len(result.factor_returns)),
            "leakage_passed": leakage.passed,
            "diagnostics": {
                "min_portfolio_n": int(result.diagnostics["min_portfolio_n"].min()),
                "months_below_min_width": int(result.diagnostics["below_min_width"].sum()),
                "median_eligible": int(result.diagnostics["n_eligible"].median()),
            },
        },
    )
    write_manifest(output_dir / "run_manifest.json", manifest)

    summary_factors = result.summary()["factors"]
    if not args.no_registry:
        low, high = spec.value_breakpoints
        append_registry_row(
            args.registry,
            {
                "experiment_id": spec.experiment_id,
                "created_at": manifest["created_at_utc"][:10],
                "project": "CH-3 reproduction (Liu-Stambaugh-Yuan 2019)",
                "research_question": run_meta.get("research_question", ""),
                "dataset_version": panel.data_version,
                "train_end": "",
                "validation_period": "",
                "test_period": f"{spec.sample_start}/{spec.sample_end}",
                "holdout_used": "no",
                "features": (
                    "monthly total market cap; E/P = TTM net profit excl. non-recurring / "
                    "total ME; B/M; listing age; 12m and 1m trading-day counts"
                ),
                "model": "2x3 independent sorts (CH-3), plus the B/M FF-3 control",
                "hyperparameters": (
                    f"drop smallest {spec.exclude_smallest_fraction:.0%} by market cap; "
                    f"size at median of the remainder; E/P {low:.0%}/{high - low:.0%}/"
                    f"{1 - high:.0%} top=value; value-weighted; monthly"
                ),
                "seed": str(spec.seed),
                "cost_config": "not applicable: factor returns, no execution assumed",
                "portfolio_config": (
                    "value-weighted; formed at month end t, returns realised in t+1"
                ),
                "code_commit": manifest["code_commit"],
                "status": "completed",
                "result_path": str(output_dir.relative_to(ROOT))
                if str(output_dir).startswith(str(ROOT))
                else str(output_dir),
                "decision": "pending review",
                "notes": run_meta.get("registry_notes", ""),
            },
        )

    print(f"\nfactor months: {len(result.factor_returns)}")
    for factor, stats in summary_factors.items():
        if stats.get("observations"):
            print(
                f"  {factor:7s} mean {stats['mean_monthly'] * 100:6.3f}%/month  "
                f"sd {stats['std_monthly'] * 100:6.3f}%  t {stats['t_stat'] or float('nan'):5.2f}"
            )
    if not alignment.empty:
        print(f"\nalignment against {author_path.name} (target r >= {target_corr}):")
        for row in alignment.to_dict("records"):
            if pd.notna(row.get("correlation")):
                print(
                    f"  {row['factor']:7s} r={row['correlation']:.4f}  "
                    f"mean gap {row['mean_gap_pct']:+.3f}pp  sd gap {row['std_gap_pct']:+.3f}pp  "
                    f"[{row['status']}]"
                )
            else:
                print(f"  {row['factor']:7s} {row['status']}")
    print(f"\noutputs: {output_dir}")

    if args.require_alignment:
        core = {row["factor"]: row["status"] for row in alignment.to_dict("records")}
        missed = [name for name in ("MKT", "SMB", "VMG") if core.get(name) != "meets target"]
        if missed:
            print(
                f"\n{len(missed)} factor(s) below the correlation target",
                file=sys.stderr,
            )
            return 1
    if not leakage.passed:
        print(
            "\nleakage checks FAILED. The run finished only because --allow-leakage "
            "was given; the report and the manifest record the override.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeakageError as error:
        print(f"\n{error}", file=sys.stderr)
        raise SystemExit(3) from error
