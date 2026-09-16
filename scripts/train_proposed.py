"""Run the canonical strict-v2 Proposed method from the audited runtime."""

from __future__ import annotations

import argparse
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

from paper.teacher.KBS_codex_revision_20260905.runs import (  # noqa: E402
    run_strict_v2_proposed as strict,
)


DATASET_NAMES = (
    "DJIA30_2026",
    "SP500_TOP50_EX_PLTR_2026",
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the frozen strict-v2 full Proposed method."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing one strict-v2 subdirectory per dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "proposed",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "reports" / "proposed",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Evaluate the frozen final test only after every requested learner is fitted.",
    )
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    missing = [name for name in args.datasets if not (data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing strict-v2 dataset directories: {missing}")
    if args.seed_start < 1 or args.seed_end < args.seed_start:
        raise ValueError("Require 1 <= seed-start <= seed-end")
    if args.smoke and args.allow_final_test:
        raise ValueError("Smoke runs must keep final test sealed")

    strict.DATA_ROOT = data_root
    strict.DATASETS = {name: data_root / name for name in DATASET_NAMES}
    strict.OUTPUT_ROOT = args.output_root.resolve()
    strict.REPORT_ROOT = args.report_root.resolve()


def main() -> None:
    args = parse_args()
    configure(args)
    forwarded = [
        str(strict.__file__),
        "--variant",
        "full_hhi",
        "--datasets",
        *args.datasets,
        "--seed_start",
        str(args.seed_start),
        "--seed_end",
        str(args.seed_end),
        "--learner_count",
        "1" if args.smoke else "5",
        "--device",
        args.device,
    ]
    if args.smoke:
        forwarded.append("--smoke")
    if args.allow_final_test:
        forwarded.append("--allow_final_test")
    sys.argv = forwarded
    strict.main()


if __name__ == "__main__":
    main()
