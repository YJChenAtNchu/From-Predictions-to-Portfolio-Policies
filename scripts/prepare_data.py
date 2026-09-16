"""Build one strict-v2 next-close dataset from a locked universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "prediction_to_portfolio_policies"
    / "runtime"
)
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from scripts import build_execution_t1_t6_datasets_20260902 as base  # noqa: E402
from scripts import build_execution_t1_t6_v2_datasets_20260905 as strict_builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adjusted-close[t+6]/adjusted-close[t+1] samples."
    )
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-name-style", choices=("us", "taiwan"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = base.DatasetSpec(
        key=args.dataset_key,
        source_root=args.source_root.resolve(),
        raw_root=args.raw_root.resolve(),
        raw_name_style=args.raw_name_style,
    )
    schedule = strict_builder.canonical_information_dates(spec)
    summary = base.build_dataset(
        spec,
        scheduled_dates=schedule,
        output_parent=args.output_root.resolve(),
    )
    summary.update(
        {
            "protocol_id": strict_builder.PROTOCOL_ID,
            "schedule_definition": (
                "fixed five-session cadence on the locked-universe common trading calendar"
            ),
            "schedule_anchor": schedule[0].date().isoformat(),
            "schedule_source": "reconstructed from common trading dates",
        }
    )
    summary_path = args.output_root.resolve() / args.dataset_key / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
