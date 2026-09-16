from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import torch_data_formal_split as legacy


EXPECTED_TARGET = "adjusted_close[t+6] / adjusted_close[t+1]"


def _load_dates(root: Path, split: str) -> pd.DataFrame:
    path = root / f"DJIA_{split}_sample_dates.csv"
    frame = pd.read_csv(path)
    required = {
        "sample_index",
        "information_end_date",
        "execution_date",
        "target_end_date",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing execution-protocol columns: {missing}")
    for column in (
        "input_start_date",
        "normalization_basis_date",
        "information_end_date",
        "input_end_date",
        "execution_date",
        "target_end_date",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    chronology = (
        (frame["information_end_date"] < frame["execution_date"])
        & (frame["execution_date"] < frame["target_end_date"])
    )
    if not bool(chronology.all()):
        raise ValueError(f"Chronology failed in {path}")
    return frame


def _validate_protocol(
    root: Path,
    train_dates: pd.DataFrame,
    val_dates: pd.DataFrame,
    test_dates: pd.DataFrame | None,
) -> dict[str, bool]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    checks = {
        "audit_status_pass": summary.get("audit_status") == "PASS",
        "target_is_next_close_execution": summary.get("target_definition") == EXPECTED_TARGET,
        "normalization_has_no_fallback": summary.get("normalization_fallback")
        == "none; samples without a prior close are excluded",
        "outer_train_exit_lte_2020": bool(
            (train_dates["target_end_date"] <= pd.Timestamp("2020-12-31")).all()
        ),
        "outer_val_interval_inside_2021_2023": bool(
            (
                (val_dates["execution_date"] >= pd.Timestamp("2021-01-01"))
                & (val_dates["target_end_date"] <= pd.Timestamp("2023-12-31"))
            ).all()
        ),
    }
    if test_dates is not None:
        checks["final_test_interval_inside_2024_2025"] = bool(
            (
                (test_dates["execution_date"] >= pd.Timestamp("2024-01-01"))
                & (test_dates["target_end_date"] <= pd.Timestamp("2025-12-31"))
            ).all()
        )
    if not all(checks.values()):
        raise ValueError(f"Execution protocol audit failed for {root}: {checks}")
    return checks


def _load_xy(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(root / f"DJIA_{split}_asset_states.npy").astype(np.float32)
    y = np.load(root / f"DJIA_{split}_asset_rors.npy").astype(np.float32)
    if x.ndim != 4 or y.ndim != 2 or x.shape[:2] != y.shape:
        raise ValueError(f"Bad {split} shapes: X={x.shape}, y={y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in {root} {split}")
    return x, y


def load_proposed_execution_data(
    root: str | Path,
    inner_train_val_split: float = 0.9,
    device: str | torch.device | None = None,
    use_signature: bool = False,
    signature_level: int = 2,
    allow_final_test: bool = False,
) -> legacy.FormalProposedDataBundle:
    root_path = Path(root)
    device_obj = torch.device(device) if device is not None else None
    sector_groups, symbols = legacy._read_order_csv(root_path / "order.csv")
    x_outer_train, y_outer_train = _load_xy(root_path, "outer_train")
    x_outer_val, y_outer_val = _load_xy(root_path, "outer_val")
    train_dates = _load_dates(root_path, "outer_train")
    val_dates = _load_dates(root_path, "outer_val")

    if x_outer_train.shape[1] != len(symbols) or x_outer_val.shape[1] != len(symbols):
        raise ValueError("Asset count does not match order.csv")
    original_seq_len = int(x_outer_train.shape[2])
    original_features = int(x_outer_train.shape[3])

    test_dates: pd.DataFrame | None = None
    if allow_final_test:
        x_final_test, y_final_test = _load_xy(root_path, "final_test")
        test_dates = _load_dates(root_path, "final_test")
    else:
        x_final_test = np.empty((0, len(symbols), original_seq_len, original_features), dtype=np.float32)
        y_final_test = np.empty((0, len(symbols)), dtype=np.float32)
    protocol_checks = _validate_protocol(root_path, train_dates, val_dates, test_dates)

    if use_signature:
        x_outer_train = legacy._signature_assetwise(x_outer_train, signature_level)
        x_outer_val = legacy._signature_assetwise(x_outer_val, signature_level)
        if allow_final_test:
            x_final_test = legacy._signature_assetwise(x_final_test, signature_level)

    n_inner_train = int(len(x_outer_train) * inner_train_val_split)
    if n_inner_train <= 0 or n_inner_train >= len(x_outer_train):
        raise ValueError(f"Invalid inner split for n={len(x_outer_train)}")

    train_sectors = legacy._sector_split(x_outer_train, sector_groups)
    val_sectors = legacy._sector_split(x_outer_val, sector_groups)
    test_sectors = legacy._sector_split(x_final_test, sector_groups)
    inner_train_sectors = [x[:n_inner_train] for x in train_sectors]
    inner_val_sectors = [x[n_inner_train:] for x in train_sectors]

    y_train_reg_all, y_train_rate_all, y_train_cls_all = legacy._make_reg_rate_cls(
        y_outer_train, sector_groups
    )
    y_outer_val_reg, y_outer_val_rate, y_outer_val_cls = legacy._make_reg_rate_cls(
        y_outer_val, sector_groups
    )
    y_final_reg, y_final_rate, y_final_cls = legacy._make_reg_rate_cls(
        y_final_test, sector_groups
    )
    dataset_summary = json.loads((root_path / "dataset_summary.json").read_text(encoding="utf-8"))
    inner_train_dates = train_dates.iloc[:n_inner_train].reset_index(drop=True)
    inner_val_dates = train_dates.iloc[n_inner_train:].reset_index(drop=True)

    def date_summary(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"n": 0, "sealed": True}
        return {
            "n": int(len(frame)),
            "information_first": frame["information_end_date"].min().date().isoformat(),
            "information_last": frame["information_end_date"].max().date().isoformat(),
            "execution_first": frame["execution_date"].min().date().isoformat(),
            "execution_last": frame["execution_date"].max().date().isoformat(),
            "target_first": frame["target_end_date"].min().date().isoformat(),
            "target_last": frame["target_end_date"].max().date().isoformat(),
        }

    meta: dict[str, Any] = {
        "root": str(root_path.resolve()),
        "dataset_name": dataset_summary["dataset_name"],
        "execution_protocol": "information close[t] -> execute close[t+1] -> exit close[t+6]",
        "target_definition": EXPECTED_TARGET,
        "final_test_access": "enabled" if allow_final_test else "sealed_not_loaded",
        "inner_train_val_split": float(inner_train_val_split),
        "n_outer_train_raw": int(len(x_outer_train)),
        "n_train": int(n_inner_train),
        "n_val": int(len(x_outer_train) - n_inner_train),
        "n_outer_val": int(len(x_outer_val)),
        "n_final_test": int(len(x_final_test)),
        "n_assets": int(len(symbols)),
        "seq_len": int(x_outer_train.shape[2]),
        "features_per_asset": int(x_outer_train.shape[3]),
        "original_seq_len": original_seq_len,
        "original_features_per_asset": original_features,
        "use_signature": bool(use_signature),
        "signature_level": int(signature_level) if use_signature else None,
        "sector_groups": sector_groups,
        "sector_sizes": [len(group) for group in sector_groups],
        "symbols": symbols,
        "y_definition": "close[t+6] / close[t+1]; >1 up, <=1 down",
        "split_by": "complete execution-to-exit interval",
        "protocol_checks": protocol_checks,
        "date_summaries": {
            "inner_train": date_summary(inner_train_dates),
            "inner_validation": date_summary(inner_val_dates),
            "outer_train": date_summary(train_dates),
            "outer_validation": date_summary(val_dates),
            "final_test": date_summary(test_dates) if test_dates is not None else {"sealed": True},
        },
        "dataset_summary": dataset_summary,
    }

    return legacy.FormalProposedDataBundle(
        x_train_list=legacy._to_tensor_list(inner_train_sectors, device_obj),
        x_val_list=legacy._to_tensor_list(inner_val_sectors, device_obj),
        x_outer_val_list=legacy._to_tensor_list(val_sectors, device_obj),
        x_final_test_list=legacy._to_tensor_list(test_sectors, device_obj),
        y_train_reg=legacy._to_tensor(y_train_reg_all[:n_inner_train], device_obj),
        y_val_reg=legacy._to_tensor(y_train_reg_all[n_inner_train:], device_obj),
        y_outer_val_reg=legacy._to_tensor(y_outer_val_reg, device_obj),
        y_final_test_reg=legacy._to_tensor(y_final_reg, device_obj),
        y_train_rate=legacy._to_tensor(y_train_rate_all[:n_inner_train], device_obj),
        y_val_rate=legacy._to_tensor(y_train_rate_all[n_inner_train:], device_obj),
        y_outer_val_rate=legacy._to_tensor(y_outer_val_rate, device_obj),
        y_final_test_rate=legacy._to_tensor(y_final_rate, device_obj),
        y_train_cls=legacy._to_tensor(y_train_cls_all[:n_inner_train], device_obj),
        y_val_cls=legacy._to_tensor(y_train_cls_all[n_inner_train:], device_obj),
        y_outer_val_cls=legacy._to_tensor(y_outer_val_cls, device_obj),
        y_final_test_cls=legacy._to_tensor(y_final_cls, device_obj),
        sector_groups=sector_groups,
        symbols=symbols,
        meta=meta,
    )
