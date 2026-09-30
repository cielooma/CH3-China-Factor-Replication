"""Generate the synthetic CH-3 panel.

    python scripts/generate_synthetic_ch3_panel.py

Deterministic: the same seed always produces byte-identical files.  The
generated CSVs are not committed — they are large and fully reproducible from
this script, which is the entry point that matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_research.synthetic import (  # noqa: E402
    SyntheticPanelSpec,
    simulate_ch3_panel,
    write_panel_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--stocks", type=int, default=900)
    parser.add_argument("--start", default="2000-01")
    parser.add_argument("--end", default="2016-12")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "synthetic_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SyntheticPanelSpec(
        seed=args.seed,
        n_stocks=args.stocks,
        start=args.start,
        end=args.end,
    )
    bundle = simulate_ch3_panel(spec)
    paths = write_panel_bundle(bundle, args.output)

    truth = bundle.truth
    print(
        json.dumps(
            {
                "market_rows": truth["n_market_rows"],
                "fundamental_rows": truth["n_fundamental_rows"],
                "securities": truth["n_securities"],
                "panel_span": truth["panel_span"],
                "factor_span": truth["factor_span"],
                "latent_factor_vols": truth["latent_factor_vols"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for name, path in paths.items():
        print(f"  {name:16s} {path}")
    print(
        "\nSynthetic data. It recovers the plumbing, never an investment conclusion."
    )


if __name__ == "__main__":
    main()
