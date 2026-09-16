from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PARENT = PROJECT_ROOT / "data" / "execution_t1_t6_20260902"
REPORT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "daily_report"
    / "execution_t1_t6_data_audit_2026-09-02"
)
FEATURE_ORDER = ("Close", "Open", "High", "Low")
LOOKBACK = 15
HOLDING_DAYS = 5
STRIDE = 5


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_root: Path
    raw_root: Path
    raw_name_style: str


DATASETS = {
    "DJIA30_2026": DatasetSpec(
        key="DJIA30_2026",
        source_root=PROJECT_ROOT
        / "data"
        / "djia30_2026_seq15_tar5_stride5_ratio_val2021_2023_test2024_2025",
        raw_root=PROJECT_ROOT / "data" / "raw_yfinance_djia30_2026",
        raw_name_style="us",
    ),
    "SP500_TOP50_EX_PLTR_2026": DatasetSpec(
        key="SP500_TOP50_EX_PLTR_2026",
        source_root=PROJECT_ROOT
        / "data"
        / "sp500_top50_ex_pltr_2026_seq15_tar5_stride5_ratio_val2021_2023_test2024_2025",
        raw_root=PROJECT_ROOT / "data" / "raw_yfinance_sp500_top50_ex_pltr_2026",
        raw_name_style="us",
    ),
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED": DatasetSpec(
        key="TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED",
        source_root=PROJECT_ROOT
        / "data"
        / "taiwan50_0050_ex_late_history_2026_seq15_tar5_stride5_ratio_prevclose_adjusted_val2021_2023_test2024_2025",
        raw_root=PROJECT_ROOT
        / "data"
        / "raw_yfinance_taiwan50_0050_2026_prevclose_adjusted",
        raw_name_style="taiwan",
    ),
}


def raw_path(spec: DatasetSpec, symbol: str) -> Path:
    if spec.raw_name_style == "taiwan":
        stem = symbol.replace(".", "_").replace("-", "_")
    else:
        stem = symbol.replace("BRK.B", "BRK-B")
    return spec.raw_root / f"{stem}.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", *FEATURE_ORDER}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    out = frame[["Date", *FEATURE_ORDER]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    for column in FEATURE_ORDER:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} has no valid adjusted OHLC rows")
    return out


def align_common_dates(frames: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    common: set[pd.Timestamp] | None = None
    for frame in frames.values():
        dates = set(frame["Date"])
        common = dates if common is None else common.intersection(dates)
    if not common:
        raise ValueError("No common dates across the locked universe")
    dates = pd.DatetimeIndex(sorted(common))
    aligned = {
        symbol: frame.set_index("Date").loc[dates].reset_index()
        for symbol, frame in frames.items()
    }
    return dates, aligned


def source_information_dates(source_root: Path) -> list[pd.Timestamp]:
    chunks = []
    for split in ("outer_train", "outer_val", "final_test"):
        path = source_root / f"DJIA_{split}_sample_dates.csv"
        frame = pd.read_csv(path)
        chunks.append(pd.to_datetime(frame["input_end_date"]).dt.normalize())
    dates = pd.concat(chunks, ignore_index=True).drop_duplicates().sort_values()
    return list(dates)


def split_name(execution_date: pd.Timestamp, target_end_date: pd.Timestamp) -> str | None:
    if target_end_date <= pd.Timestamp("2020-12-31"):
        return "outer_train"
    if (
        execution_date >= pd.Timestamp("2021-01-01")
        and target_end_date <= pd.Timestamp("2023-12-31")
    ):
        return "outer_val"
    if (
        execution_date >= pd.Timestamp("2024-01-01")
        and target_end_date <= pd.Timestamp("2025-12-31")
    ):
        return "final_test"
    return None


def build_dataset(
    spec: DatasetSpec,
    scheduled_dates: list[pd.Timestamp] | None = None,
    output_parent: Path | None = None,
) -> dict[str, Any]:
    order = pd.read_csv(spec.source_root / "order.csv").sort_values("order").reset_index(drop=True)
    symbols = order["Ticker"].astype(str).tolist()
    paths = {symbol: raw_path(spec, symbol) for symbol in symbols}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files for {spec.key}: {missing}")

    frames = {symbol: read_raw(paths[symbol]) for symbol in symbols}
    common_dates, aligned = align_common_dates(frames)
    date_to_index = {date: index for index, date in enumerate(common_dates)}
    if scheduled_dates is None:
        scheduled_dates = source_information_dates(spec.source_root)

    prices = {
        symbol: {
            column: aligned[symbol][column].to_numpy(dtype=np.float64)
            for column in FEATURE_ORDER
        }
        for symbol in symbols
    }
    samples: dict[str, dict[str, list[Any]]] = {
        split: {
            "states": [],
            "rors": [],
            "information_closes": [],
            "execution_closes": [],
            "target_closes": [],
            "dates": [],
        }
        for split in ("outer_train", "outer_val", "final_test")
    }
    excluded: list[dict[str, str]] = []

    for information_date in scheduled_dates:
        if information_date not in date_to_index:
            raise ValueError(f"Scheduled information date {information_date.date()} is absent from common dates")
        t = date_to_index[information_date]
        window_start = t - LOOKBACK + 1
        normalization_index = window_start - 1
        execution_index = t + 1
        target_index = t + HOLDING_DAYS + 1
        if normalization_index < 0 or target_index >= len(common_dates):
            excluded.append({"information_end_date": information_date.date().isoformat(), "reason": "insufficient_context"})
            continue

        execution_date = common_dates[execution_index]
        target_end_date = common_dates[target_index]
        split = split_name(execution_date, target_end_date)
        if split is None:
            excluded.append(
                {
                    "information_end_date": information_date.date().isoformat(),
                    "execution_date": execution_date.date().isoformat(),
                    "target_end_date": target_end_date.date().isoformat(),
                    "reason": "crosses_split_boundary_or_outside_protocol",
                }
            )
            continue

        asset_states = []
        asset_rors = []
        information_closes = []
        execution_closes = []
        target_closes = []
        for symbol in symbols:
            close = prices[symbol]["Close"]
            basis = close[normalization_index]
            information_close = close[t]
            execution_close = close[execution_index]
            target_close = close[target_index]
            if min(basis, information_close, execution_close, target_close) <= 0:
                raise ValueError(f"Non-positive adjusted close for {symbol} at {information_date.date()}")
            window = np.stack(
                [prices[symbol][column][window_start : t + 1] / basis for column in FEATURE_ORDER],
                axis=-1,
            )
            asset_states.append(window.astype(np.float32))
            asset_rors.append(np.float32(target_close / execution_close))
            information_closes.append(np.float32(information_close))
            execution_closes.append(np.float32(execution_close))
            target_closes.append(np.float32(target_close))

        record = samples[split]
        record["states"].append(np.stack(asset_states, axis=0))
        record["rors"].append(np.asarray(asset_rors, dtype=np.float32))
        record["information_closes"].append(np.asarray(information_closes, dtype=np.float32))
        record["execution_closes"].append(np.asarray(execution_closes, dtype=np.float32))
        record["target_closes"].append(np.asarray(target_closes, dtype=np.float32))
        record["dates"].append(
            {
                "input_start_date": common_dates[window_start].date().isoformat(),
                "normalization_basis_date": common_dates[normalization_index].date().isoformat(),
                "information_end_date": information_date.date().isoformat(),
                "input_end_date": information_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat(),
                "target_end_date": target_end_date.date().isoformat(),
            }
        )

    output_root = (output_parent or OUTPUT_PARENT) / spec.key
    output_root.mkdir(parents=True, exist_ok=True)
    order.to_csv(output_root / "order.csv", index=False, encoding="utf-8-sig")
    stats: dict[str, Any] = {}
    for split, record in samples.items():
        arrays = {
            "asset_states": np.asarray(record["states"], dtype=np.float32),
            "asset_rors": np.asarray(record["rors"], dtype=np.float32),
            "asset_window_last_closes": np.asarray(record["information_closes"], dtype=np.float32),
            "asset_execution_closes": np.asarray(record["execution_closes"], dtype=np.float32),
            "asset_target_closes": np.asarray(record["target_closes"], dtype=np.float32),
        }
        if not arrays["asset_states"].size:
            raise ValueError(f"{spec.key} {split} is empty")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"{spec.key} {split} contains non-finite values")
        recomputed = arrays["asset_target_closes"] / arrays["asset_execution_closes"]
        max_target_error = float(np.max(np.abs(recomputed - arrays["asset_rors"])))
        if max_target_error > 2e-6:
            raise ValueError(f"{spec.key} {split} target audit failed: {max_target_error}")
        for suffix, array in arrays.items():
            np.save(output_root / f"DJIA_{split}_{suffix}.npy", array)
        dates = pd.DataFrame(record["dates"])
        dates.insert(0, "sample_index", np.arange(len(dates)))
        dates.to_csv(output_root / f"DJIA_{split}_sample_dates.csv", index=False, encoding="utf-8-sig")
        date_checks = (
            (pd.to_datetime(dates["information_end_date"]) < pd.to_datetime(dates["execution_date"]))
            & (pd.to_datetime(dates["execution_date"]) < pd.to_datetime(dates["target_end_date"]))
        )
        if not bool(date_checks.all()):
            raise ValueError(f"{spec.key} {split} chronology audit failed")
        stats[split] = {
            "n_samples": int(len(dates)),
            "states_shape": list(arrays["asset_states"].shape),
            "information_first": dates["information_end_date"].iloc[0],
            "information_last": dates["information_end_date"].iloc[-1],
            "execution_first": dates["execution_date"].iloc[0],
            "execution_last": dates["execution_date"].iloc[-1],
            "target_first": dates["target_end_date"].iloc[0],
            "target_last": dates["target_end_date"].iloc[-1],
            "max_target_recompute_error": max_target_error,
        }

    checksums = pd.DataFrame(
        [
            {"symbol": symbol, "raw_path": str(paths[symbol]), "sha256": sha256(paths[symbol])}
            for symbol in symbols
        ]
    )
    checksums.to_csv(output_root / "raw_file_checksums.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded).to_csv(output_root / "excluded_samples.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset_name": spec.key,
        "source_dataset_root": str(spec.source_root),
        "output_root": str(output_root),
        "universe_locked_from_source": True,
        "n_assets": len(symbols),
        "symbols": symbols,
        "feature_order": ["norm_close", "norm_open", "norm_high", "norm_low"],
        "raw_feature_order": list(FEATURE_ORDER),
        "price_basis": "adjusted OHLC from the existing locked raw files",
        "normalization": "all 15-day adjusted OHLC values divided by adjusted Close immediately before the input window",
        "normalization_fallback": "none; samples without a prior close are excluded",
        "information_set": "adjusted OHLC through close[t]",
        "decision_time": "after close[t]",
        "execution": "adjusted close[t+1]",
        "target_definition": "adjusted_close[t+6] / adjusted_close[t+1]",
        "target_label": "up iff target ratio > 1",
        "lookback": LOOKBACK,
        "holding_days": HOLDING_DAYS,
        "stride": STRIDE,
        "split_rule": "complete execution-to-exit interval must remain inside its chronological split",
        "outer_train": "target_end_date <= 2020-12-31",
        "outer_validation": "execution_date >= 2021-01-01 and target_end_date <= 2023-12-31",
        "final_test": "execution_date >= 2024-01-01 and target_end_date <= 2025-12-31",
        "inner_validation": "last 10% of outer_train in chronological order; checkpoint only",
        "stats": stats,
        "excluded_sample_count": len(excluded),
        "audit_status": "PASS",
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    invalid = [name for name in args.datasets if name not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {name: build_dataset(DATASETS[name]) for name in args.datasets}
    audit_rows = []
    for name, summary in summaries.items():
        for split, stats in summary["stats"].items():
            audit_rows.append(
                {
                    "dataset": name,
                    "split": split,
                    **stats,
                    "target_definition": summary["target_definition"],
                    "normalization": summary["normalization"],
                    "audit_status": summary["audit_status"],
                }
            )
    pd.DataFrame(audit_rows).to_csv(
        REPORT_ROOT / "execution_t1_t6_dataset_audit.csv", index=False, encoding="utf-8-sig"
    )
    (REPORT_ROOT / "execution_t1_t6_dataset_audit.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
