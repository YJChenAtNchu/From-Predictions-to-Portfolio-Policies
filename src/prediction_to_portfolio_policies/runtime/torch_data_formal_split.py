from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from torch_data import (
    _make_reg_rate_cls,
    _read_order_csv,
    _sector_split,
    _signature_assetwise,
    _tensor_summary,
    _to_tensor,
    _to_tensor_list,
)


@dataclass
class FormalProposedDataBundle:
    x_train_list: List[torch.Tensor]
    x_val_list: List[torch.Tensor]
    x_outer_val_list: List[torch.Tensor]
    x_final_test_list: List[torch.Tensor]
    y_train_reg: torch.Tensor
    y_val_reg: torch.Tensor
    y_outer_val_reg: torch.Tensor
    y_final_test_reg: torch.Tensor
    y_train_rate: torch.Tensor
    y_val_rate: torch.Tensor
    y_outer_val_rate: torch.Tensor
    y_final_test_rate: torch.Tensor
    y_train_cls: torch.Tensor
    y_val_cls: torch.Tensor
    y_outer_val_cls: torch.Tensor
    y_final_test_cls: torch.Tensor
    sector_groups: List[List[int]]
    symbols: List[str]
    meta: Dict[str, object]


def _load_dates(root: Path, prefix: str) -> pd.DataFrame:
    path = root / f"DJIA_{prefix}_sample_dates.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required date file: {path}")
    dates = pd.read_csv(path, parse_dates=["input_end_date", "target_end_date"])
    required = {"sample_index", "input_end_date", "target_end_date"}
    missing = required.difference(dates.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return dates


def _date_summary(dates: pd.DataFrame) -> Dict[str, object]:
    return {
        "n_samples": int(len(dates)),
        "input_end_first": dates["input_end_date"].min().date().isoformat(),
        "input_end_last": dates["input_end_date"].max().date().isoformat(),
        "target_end_first": dates["target_end_date"].min().date().isoformat(),
        "target_end_last": dates["target_end_date"].max().date().isoformat(),
    }


def _assert_target_date_rules(
    outer_train_dates: pd.DataFrame,
    outer_val_dates: pd.DataFrame,
    final_test_dates: pd.DataFrame,
) -> Dict[str, bool]:
    checks = {
        "outer_train_target_end_lte_2020_12_31": bool(
            (outer_train_dates["target_end_date"] <= pd.Timestamp("2020-12-31")).all()
        ),
        "outer_val_target_end_2021_2023": bool(
            (
                (outer_val_dates["target_end_date"] >= pd.Timestamp("2021-01-01"))
                & (outer_val_dates["target_end_date"] <= pd.Timestamp("2023-12-31"))
            ).all()
        ),
        "final_test_target_end_2024_2025": bool(
            (
                (final_test_dates["target_end_date"] >= pd.Timestamp("2024-01-01"))
                & (final_test_dates["target_end_date"] <= pd.Timestamp("2025-12-31"))
            ).all()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Formal split target-date rule failed: {checks}")
    return checks


def load_proposed_djia_formal_data(
    root: str | Path,
    inner_train_val_split: float = 0.9,
    device: str | torch.device | None = None,
    use_signature: bool = False,
    signature_level: int = 2,
) -> FormalProposedDataBundle:
    root_path = Path(root)
    device_obj = torch.device(device) if device is not None else None

    paths = {
        "order": root_path / "order.csv",
        "outer_train_x": root_path / "DJIA_outer_train_asset_states.npy",
        "outer_train_y": root_path / "DJIA_outer_train_asset_rors.npy",
        "outer_val_x": root_path / "DJIA_outer_val_asset_states.npy",
        "outer_val_y": root_path / "DJIA_outer_val_asset_rors.npy",
        "final_test_x": root_path / "DJIA_final_test_asset_states.npy",
        "final_test_y": root_path / "DJIA_final_test_asset_rors.npy",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    sector_groups, symbols = _read_order_csv(paths["order"])
    x_outer_train_raw = np.load(paths["outer_train_x"])
    y_outer_train_raw = np.load(paths["outer_train_y"])
    x_outer_val_raw = np.load(paths["outer_val_x"])
    y_outer_val_raw = np.load(paths["outer_val_y"])
    x_final_test_raw = np.load(paths["final_test_x"])
    y_final_test_raw = np.load(paths["final_test_y"])

    for name, x_raw, y_raw in [
        ("outer_train", x_outer_train_raw, y_outer_train_raw),
        ("outer_val", x_outer_val_raw, y_outer_val_raw),
        ("final_test", x_final_test_raw, y_final_test_raw),
    ]:
        if x_raw.ndim != 4 or y_raw.ndim != 2:
            raise ValueError(f"{name} bad shape: X={x_raw.shape}, y={y_raw.shape}")
        if x_raw.shape[1] != len(symbols) or y_raw.shape[1] != len(symbols):
            raise ValueError(f"{name} asset mismatch: X={x_raw.shape}, y={y_raw.shape}, symbols={len(symbols)}")

    original_seq_len = int(x_outer_train_raw.shape[2])
    original_features_per_asset = int(x_outer_train_raw.shape[3])
    if use_signature:
        x_outer_train_raw = _signature_assetwise(x_outer_train_raw, signature_level)
        x_outer_val_raw = _signature_assetwise(x_outer_val_raw, signature_level)
        x_final_test_raw = _signature_assetwise(x_final_test_raw, signature_level)

    n_inner_train = int(x_outer_train_raw.shape[0] * inner_train_val_split)
    if n_inner_train <= 0 or n_inner_train >= x_outer_train_raw.shape[0]:
        raise ValueError(
            f"Invalid inner_train_val_split={inner_train_val_split} for outer_train n={x_outer_train_raw.shape[0]}"
        )

    outer_train_sectors = _sector_split(x_outer_train_raw, sector_groups)
    outer_val_sectors = _sector_split(x_outer_val_raw, sector_groups)
    final_test_sectors = _sector_split(x_final_test_raw, sector_groups)
    inner_train_sectors = [x[:n_inner_train] for x in outer_train_sectors]
    inner_val_sectors = [x[n_inner_train:] for x in outer_train_sectors]

    y_outer_train_reg_all, y_outer_train_rate_all, y_outer_train_cls_all = _make_reg_rate_cls(
        y_outer_train_raw, sector_groups
    )
    y_outer_val_reg, y_outer_val_rate, y_outer_val_cls = _make_reg_rate_cls(y_outer_val_raw, sector_groups)
    y_final_test_reg, y_final_test_rate, y_final_test_cls = _make_reg_rate_cls(y_final_test_raw, sector_groups)

    y_train_reg = y_outer_train_reg_all[:n_inner_train]
    y_val_reg = y_outer_train_reg_all[n_inner_train:]
    y_train_rate = y_outer_train_rate_all[:n_inner_train]
    y_val_rate = y_outer_train_rate_all[n_inner_train:]
    y_train_cls = y_outer_train_cls_all[:n_inner_train]
    y_val_cls = y_outer_train_cls_all[n_inner_train:]

    outer_train_dates = _load_dates(root_path, "outer_train")
    outer_val_dates = _load_dates(root_path, "outer_val")
    final_test_dates = _load_dates(root_path, "final_test")
    split_rule_checks = _assert_target_date_rules(outer_train_dates, outer_val_dates, final_test_dates)
    inner_train_dates = outer_train_dates.iloc[:n_inner_train].reset_index(drop=True)
    inner_val_dates = outer_train_dates.iloc[n_inner_train:].reset_index(drop=True)

    dataset_summary_path = root_path / "dataset_summary.json"
    dataset_summary: Dict[str, object] = {}
    if dataset_summary_path.is_file():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))

    meta: Dict[str, object] = {
        "root": str(root_path.resolve()),
        "dataset_name": dataset_summary.get("dataset_name", root_path.name),
        "inner_train_val_split": float(inner_train_val_split),
        "n_outer_train_raw": int(x_outer_train_raw.shape[0]),
        "n_train": int(n_inner_train),
        "n_val": int(x_outer_train_raw.shape[0] - n_inner_train),
        "n_outer_val": int(x_outer_val_raw.shape[0]),
        "n_final_test": int(x_final_test_raw.shape[0]),
        "n_assets": int(len(symbols)),
        "seq_len": int(x_outer_train_raw.shape[2]),
        "features_per_asset": int(x_outer_train_raw.shape[3]),
        "use_signature": bool(use_signature),
        "signature_level": int(signature_level) if use_signature else None,
        "original_seq_len": original_seq_len,
        "original_features_per_asset": original_features_per_asset,
        "sector_groups": sector_groups,
        "sector_sizes": [len(g) for g in sector_groups],
        "symbols": symbols,
        "y_definition": "return ratio; >1 up, <=1 down",
        "split_by": "target_end_date",
        "date_summaries": {
            "inner_train": _date_summary(inner_train_dates),
            "inner_validation": _date_summary(inner_val_dates),
            "outer_train": _date_summary(outer_train_dates),
            "outer_validation": _date_summary(outer_val_dates),
            "final_test": _date_summary(final_test_dates),
        },
        "split_rule_checks": split_rule_checks,
        "dataset_summary": dataset_summary,
    }

    return FormalProposedDataBundle(
        x_train_list=_to_tensor_list(inner_train_sectors, device_obj),
        x_val_list=_to_tensor_list(inner_val_sectors, device_obj),
        x_outer_val_list=_to_tensor_list(outer_val_sectors, device_obj),
        x_final_test_list=_to_tensor_list(final_test_sectors, device_obj),
        y_train_reg=_to_tensor(y_train_reg, device_obj),
        y_val_reg=_to_tensor(y_val_reg, device_obj),
        y_outer_val_reg=_to_tensor(y_outer_val_reg, device_obj),
        y_final_test_reg=_to_tensor(y_final_test_reg, device_obj),
        y_train_rate=_to_tensor(y_train_rate, device_obj),
        y_val_rate=_to_tensor(y_val_rate, device_obj),
        y_outer_val_rate=_to_tensor(y_outer_val_rate, device_obj),
        y_final_test_rate=_to_tensor(y_final_test_rate, device_obj),
        y_train_cls=_to_tensor(y_train_cls, device_obj),
        y_val_cls=_to_tensor(y_val_cls, device_obj),
        y_outer_val_cls=_to_tensor(y_outer_val_cls, device_obj),
        y_final_test_cls=_to_tensor(y_final_test_cls, device_obj),
        sector_groups=sector_groups,
        symbols=symbols,
        meta=meta,
    )


def build_formal_data_check(bundle: FormalProposedDataBundle) -> Dict[str, object]:
    arrays: List[Dict[str, object]] = []
    for split_name, xs in [
        ("x_train", bundle.x_train_list),
        ("x_inner_val", bundle.x_val_list),
        ("x_outer_val", bundle.x_outer_val_list),
        ("x_final_test", bundle.x_final_test_list),
    ]:
        for i, x in enumerate(xs, start=1):
            arrays.append(_tensor_summary(f"{split_name}_sector_{i}", x))

    arrays.extend(
        [
            _tensor_summary("y_train_reg", bundle.y_train_reg),
            _tensor_summary("y_val_reg", bundle.y_val_reg),
            _tensor_summary("y_outer_val_reg", bundle.y_outer_val_reg),
            _tensor_summary("y_final_test_reg", bundle.y_final_test_reg),
            _tensor_summary("y_train_cls", bundle.y_train_cls),
            _tensor_summary("y_val_cls", bundle.y_val_cls),
            _tensor_summary("y_outer_val_cls", bundle.y_outer_val_cls),
            _tensor_summary("y_final_test_cls", bundle.y_final_test_cls),
        ]
    )

    n_assets = int(bundle.meta["n_assets"])
    checks = {
        "has_five_sectors": len(bundle.x_train_list) == 5,
        "outer_train_count_is_601": int(bundle.meta["n_outer_train_raw"]) == 601,
        "outer_val_count_is_100": bundle.y_outer_val_reg.shape[0] == 100,
        "final_test_count_is_151": bundle.y_final_test_reg.shape[0] == 151,
        "y_reg_shape_ok": list(bundle.y_train_reg.shape[1:]) == [1, n_assets],
        "y_cls_shape_ok": list(bundle.y_train_cls.shape[1:]) == [1, n_assets, 2],
        "sector_feature_dims_sum_to_assets_times_features": sum(x.shape[-1] for x in bundle.x_train_list)
        == n_assets * int(bundle.meta["features_per_asset"]),
        "all_split_lengths_match": (
            all(x.shape[0] == bundle.y_train_reg.shape[0] for x in bundle.x_train_list)
            and all(x.shape[0] == bundle.y_val_reg.shape[0] for x in bundle.x_val_list)
            and all(x.shape[0] == bundle.y_outer_val_reg.shape[0] for x in bundle.x_outer_val_list)
            and all(x.shape[0] == bundle.y_final_test_reg.shape[0] for x in bundle.x_final_test_list)
        ),
        "cls_one_hot_sum_is_one": bool(torch.allclose(bundle.y_train_cls.sum(dim=-1), torch.ones_like(bundle.y_train_reg))),
        **bundle.meta["split_rule_checks"],
    }
    return {
        "meta": bundle.meta,
        "checks": checks,
        "arrays": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check formal DJIA30 2026 Torch data loading.")
    parser.add_argument(
        "--root",
        default="data/djia30_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
    )
    parser.add_argument("--inner_train_val_split", type=float, default=0.9)
    parser.add_argument("--use_signature", action="store_true")
    parser.add_argument("--signature_level", type=int, default=2)
    parser.add_argument(
        "--out",
        default="outputs/torch_proposed/formal_split_data_check_0603.json",
    )
    args = parser.parse_args()

    bundle = load_proposed_djia_formal_data(
        root=args.root,
        inner_train_val_split=args.inner_train_val_split,
        use_signature=args.use_signature,
        signature_level=args.signature_level,
    )
    report = build_formal_data_check(bundle)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    print(json.dumps(report["meta"]["date_summaries"], indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
