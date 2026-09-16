from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_execution_t1_t6_datasets_20260902 as base


PROTOCOL_ID = "nextclose_t1_t6_stride5_v2_20260905"
OUTPUT_PARENT = PROJECT_ROOT / "data" / "execution_t1_t6_v2_20260905"
REPORT_ROOT = PROJECT_ROOT / "outputs" / "daily_report" / "execution_t1_t6_v2_data_audit_2026-09-05"


def canonical_information_dates(spec: base.DatasetSpec) -> list[pd.Timestamp]:
    order = pd.read_csv(spec.source_root / "order.csv").sort_values("order").reset_index(drop=True)
    symbols = order["Ticker"].astype(str).tolist()
    frames = {symbol: base.read_raw(base.raw_path(spec, symbol)) for symbol in symbols}
    common_dates, _ = base.align_common_dates(frames)

    archived_schedule = base.source_information_dates(spec.source_root)
    if not archived_schedule:
        raise ValueError(f"No archived schedule is available to anchor {spec.key}")
    first_information_date = archived_schedule[0]
    if first_information_date not in common_dates:
        raise ValueError(f"Cadence anchor {first_information_date.date()} is absent from {spec.key}")

    first_index = common_dates.get_loc(first_information_date)
    return [
        common_dates[index]
        for index in range(first_index, len(common_dates), base.STRIDE)
    ]


def build_dataset(spec: base.DatasetSpec) -> dict[str, object]:
    schedule = canonical_information_dates(spec)
    summary = base.build_dataset(
        spec,
        scheduled_dates=schedule,
        output_parent=OUTPUT_PARENT,
    )
    summary.update(
        {
            "protocol_id": PROTOCOL_ID,
            "schedule_definition": (
                "fixed five-session cadence on the locked universe common trading calendar, "
                "anchored at the first archived information date"
            ),
            "schedule_anchor": schedule[0].date().isoformat(),
            "schedule_source": "reconstructed from common trading dates; archived split gaps are not inherited",
        }
    )
    output_root = OUTPUT_PARENT / spec.key
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(base.DATASETS), default=sorted(base.DATASETS))
    args = parser.parse_args()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {key: build_dataset(base.DATASETS[key]) for key in args.datasets}
    report = {
        "protocol_id": PROTOCOL_ID,
        "output_parent": str(OUTPUT_PARENT),
        "datasets": summaries,
    }
    (REPORT_ROOT / "dataset_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
