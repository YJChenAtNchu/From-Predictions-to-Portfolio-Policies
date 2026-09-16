from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import iisignature  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    iisignature = None


@dataclass
class ProposedDataBundle:
    x_train_list: List[torch.Tensor]
    x_val_list: List[torch.Tensor]
    x_test_list: List[torch.Tensor]
    y_train_reg: torch.Tensor
    y_val_reg: torch.Tensor
    y_test_reg: torch.Tensor
    y_train_rate: torch.Tensor
    y_val_rate: torch.Tensor
    y_test_rate: torch.Tensor
    y_train_cls: torch.Tensor
    y_val_cls: torch.Tensor
    y_test_cls: torch.Tensor
    sector_groups: List[List[int]]
    symbols: List[str]
    meta: Dict[str, object]


def _read_order_csv(order_path: Path) -> tuple[List[List[int]], List[str]]:
    groups: Dict[int, List[int]] = {}
    order_to_symbol: Dict[int, str] = {}
    with order_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Ticker", "index", "order"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{order_path} missing columns: {sorted(missing)}")
        for row in reader:
            sector_id = int(row["index"])
            order_idx = int(row["order"])
            groups.setdefault(sector_id, []).append(order_idx)
            order_to_symbol[order_idx] = row["Ticker"]

    sector_groups = [sorted(groups[k]) for k in sorted(groups)]
    symbols = [order_to_symbol[i] for i in sorted(order_to_symbol)]
    return sector_groups, symbols


def _signature_level2_assetwise(x_raw: np.ndarray) -> np.ndarray:
    """Asset-wise path signature up to level 2.

    This mirrors the active TensorFlow preprocessing idea:
    (N, A, T, F) -> (N, 1, A * siglen(F, 2)).

    For a piecewise-linear path, level-1 is the total increment and level-2 is
    accumulated with Chen's identity over consecutive increments.
    """
    if x_raw.ndim != 4:
        raise ValueError(f"Expected X shape (N,A,T,F), got {x_raw.shape}")
    n, n_assets, seq_len, n_features = x_raw.shape
    if seq_len < 2:
        raise ValueError("Signature transform needs at least two time points")

    dx = np.diff(x_raw.astype(np.float64), axis=2)  # (N,A,T-1,F)
    level1 = dx.sum(axis=2)  # (N,A,F)
    level2 = np.zeros((n, n_assets, n_features, n_features), dtype=np.float64)
    prefix = np.zeros((n, n_assets, n_features), dtype=np.float64)
    for step in range(dx.shape[2]):
        d = dx[:, :, step, :]
        level2 += 0.5 * np.einsum("naf,nag->nafg", d, d)
        level2 += np.einsum("naf,nag->nafg", prefix, d)
        prefix += d

    sig = np.concatenate([level1, level2.reshape(n, n_assets, n_features * n_features)], axis=-1)
    return sig[:, :, None, :].astype(np.float32)


def _signature_assetwise(x_raw: np.ndarray, level: int) -> np.ndarray:
    """Asset-wise signature transform.

    The TensorFlow proposed code uses the iisignature package.  We use the same
    package when it is available; otherwise we keep the previous level-2
    fallback so existing lightweight environments still work.
    """
    if x_raw.ndim != 4:
        raise ValueError(f"Expected X shape (N,A,T,F), got {x_raw.shape}")
    if iisignature is None:
        if int(level) != 2:
            raise ImportError("iisignature is required for signature_level other than 2")
        return _signature_level2_assetwise(x_raw)

    n, n_assets, _, n_features = x_raw.shape
    sig_len = int(iisignature.siglength(n_features, int(level)))
    out = np.zeros((n, n_assets, sig_len), dtype=np.float32)
    for sample_idx in range(n):
        for asset_idx in range(n_assets):
            out[sample_idx, asset_idx] = iisignature.sig(
                x_raw[sample_idx, asset_idx].astype(np.float64),
                int(level),
            )
    return out[:, :, None, :].astype(np.float32)


def _sector_split(x_raw: np.ndarray, sector_groups: List[List[int]]) -> List[np.ndarray]:
    if x_raw.ndim != 4:
        raise ValueError(f"Expected X shape (N,A,T,F), got {x_raw.shape}")
    outputs: List[np.ndarray] = []
    for asset_idx in sector_groups:
        sub = x_raw[:, asset_idx, :, :]  # (N, A_sector, T, F)
        n, n_assets, seq_len, n_features = sub.shape
        sector_x = sub.transpose(0, 2, 1, 3).reshape(n, seq_len, n_assets * n_features)
        outputs.append(sector_x.astype(np.float32))
    return outputs


def _make_reg_rate_cls(y_raw: np.ndarray, sector_groups: List[List[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y_raw.ndim != 2:
        raise ValueError(f"Expected y shape (N,A), got {y_raw.shape}")
    ordered_idx = [idx for group in sector_groups for idx in group]
    y_ordered = y_raw[:, ordered_idx]
    y_reg = y_ordered[:, None, :].astype(np.float32)
    y_rate = (y_reg - 1.0).astype(np.float32)

    down = (y_reg <= 1.0).astype(np.float32)
    up = (y_reg > 1.0).astype(np.float32)
    y_cls = np.stack([down, up], axis=-1).astype(np.float32)  # (N,1,A,2)
    return y_reg, y_rate, y_cls


def _to_tensor_list(xs: List[np.ndarray], device: Optional[torch.device] = None) -> List[torch.Tensor]:
    return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in xs]


def _to_tensor(x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def load_proposed_djia_data(
    root: str | Path = ".",
    train_val_split: float = 0.9,
    device: str | torch.device | None = None,
    use_signature: bool = False,
    signature_level: int = 2,
) -> ProposedDataBundle:
    root = Path(root)
    device_obj = torch.device(device) if device is not None else None

    order_path = root / "order.csv"
    x_train_path = root / "DJIA_train_asset_states.npy"
    x_test_path = root / "DJIA_test_asset_states.npy"
    y_train_path = root / "DJIA_train_asset_rors.npy"
    y_test_path = root / "DJIA_test_asset_rors.npy"

    for path in [order_path, x_train_path, x_test_path, y_train_path, y_test_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    sector_groups, symbols = _read_order_csv(order_path)
    x_train_raw = np.load(x_train_path)
    x_test_raw = np.load(x_test_path)
    y_train_raw = np.load(y_train_path)
    y_test_raw = np.load(y_test_path)

    if x_train_raw.shape[1] != len(symbols) or x_test_raw.shape[1] != len(symbols):
        raise ValueError(
            "Asset count mismatch between X arrays and order.csv: "
            f"x_train={x_train_raw.shape}, x_test={x_test_raw.shape}, symbols={len(symbols)}"
        )
    if y_train_raw.shape[1] != len(symbols) or y_test_raw.shape[1] != len(symbols):
        raise ValueError(
            "Asset count mismatch between y arrays and order.csv: "
            f"y_train={y_train_raw.shape}, y_test={y_test_raw.shape}, symbols={len(symbols)}"
        )
    original_seq_len = int(x_train_raw.shape[2])
    original_features_per_asset = int(x_train_raw.shape[3])
    if use_signature:
        x_train_raw = _signature_assetwise(x_train_raw, signature_level)
        x_test_raw = _signature_assetwise(x_test_raw, signature_level)

    n_train = int(x_train_raw.shape[0] * train_val_split)
    if n_train <= 0 or n_train >= x_train_raw.shape[0]:
        raise ValueError(f"Invalid train_val_split={train_val_split} for n={x_train_raw.shape[0]}")

    x_train_sector_all = _sector_split(x_train_raw, sector_groups)
    x_test_sector = _sector_split(x_test_raw, sector_groups)
    x_train_sector = [x[:n_train] for x in x_train_sector_all]
    x_val_sector = [x[n_train:] for x in x_train_sector_all]

    y_train_reg_all, y_train_rate_all, y_train_cls_all = _make_reg_rate_cls(y_train_raw, sector_groups)
    y_test_reg, y_test_rate, y_test_cls = _make_reg_rate_cls(y_test_raw, sector_groups)

    y_train_reg = y_train_reg_all[:n_train]
    y_val_reg = y_train_reg_all[n_train:]
    y_train_rate = y_train_rate_all[:n_train]
    y_val_rate = y_train_rate_all[n_train:]
    y_train_cls = y_train_cls_all[:n_train]
    y_val_cls = y_train_cls_all[n_train:]

    meta: Dict[str, object] = {
        "root": str(root.resolve()),
        "train_val_split": float(train_val_split),
        "n_train_raw": int(x_train_raw.shape[0]),
        "n_test": int(x_test_raw.shape[0]),
        "n_train": int(n_train),
        "n_val": int(x_train_raw.shape[0] - n_train),
        "n_assets": int(len(symbols)),
        "seq_len": int(x_train_raw.shape[2]),
        "features_per_asset": int(x_train_raw.shape[3]),
        "use_signature": bool(use_signature),
        "signature_level": int(signature_level) if use_signature else None,
        "original_seq_len": original_seq_len,
        "original_features_per_asset": original_features_per_asset,
        "sector_groups": sector_groups,
        "sector_sizes": [len(g) for g in sector_groups],
        "symbols": symbols,
        "y_definition": "return ratio; >1 up, <=1 down",
    }

    return ProposedDataBundle(
        x_train_list=_to_tensor_list(x_train_sector, device_obj),
        x_val_list=_to_tensor_list(x_val_sector, device_obj),
        x_test_list=_to_tensor_list(x_test_sector, device_obj),
        y_train_reg=_to_tensor(y_train_reg, device_obj),
        y_val_reg=_to_tensor(y_val_reg, device_obj),
        y_test_reg=_to_tensor(y_test_reg, device_obj),
        y_train_rate=_to_tensor(y_train_rate, device_obj),
        y_val_rate=_to_tensor(y_val_rate, device_obj),
        y_test_rate=_to_tensor(y_test_rate, device_obj),
        y_train_cls=_to_tensor(y_train_cls, device_obj),
        y_val_cls=_to_tensor(y_val_cls, device_obj),
        y_test_cls=_to_tensor(y_test_cls, device_obj),
        sector_groups=sector_groups,
        symbols=symbols,
        meta=meta,
    )


def _tensor_summary(name: str, x: torch.Tensor) -> Dict[str, object]:
    x_cpu = x.detach().cpu()
    return {
        "name": name,
        "shape": list(x_cpu.shape),
        "dtype": str(x_cpu.dtype),
        "nan_count": int(torch.isnan(x_cpu).sum().item()) if x_cpu.is_floating_point() else None,
        "min": float(x_cpu.min().item()) if x_cpu.numel() and x_cpu.is_floating_point() else None,
        "mean": float(x_cpu.mean().item()) if x_cpu.numel() and x_cpu.is_floating_point() else None,
        "max": float(x_cpu.max().item()) if x_cpu.numel() and x_cpu.is_floating_point() else None,
    }


def build_data_check(bundle: ProposedDataBundle) -> Dict[str, object]:
    arrays: List[Dict[str, object]] = []
    for split_name, xs in [
        ("x_train", bundle.x_train_list),
        ("x_val", bundle.x_val_list),
        ("x_test", bundle.x_test_list),
    ]:
        for i, x in enumerate(xs, start=1):
            arrays.append(_tensor_summary(f"{split_name}_sector_{i}", x))

    arrays.extend(
        [
            _tensor_summary("y_train_reg", bundle.y_train_reg),
            _tensor_summary("y_val_reg", bundle.y_val_reg),
            _tensor_summary("y_test_reg", bundle.y_test_reg),
            _tensor_summary("y_train_rate", bundle.y_train_rate),
            _tensor_summary("y_val_rate", bundle.y_val_rate),
            _tensor_summary("y_test_rate", bundle.y_test_rate),
            _tensor_summary("y_train_cls", bundle.y_train_cls),
            _tensor_summary("y_val_cls", bundle.y_val_cls),
            _tensor_summary("y_test_cls", bundle.y_test_cls),
        ]
    )

    n_assets = int(bundle.meta["n_assets"])
    checks = {
        "has_five_sectors": len(bundle.x_train_list) == 5,
        "train_count_is_195": bundle.y_train_reg.shape[0] == 195,
        "val_count_is_22": bundle.y_val_reg.shape[0] == 22,
        "test_count_is_32": bundle.y_test_reg.shape[0] == 32,
        "y_reg_shape_ok": list(bundle.y_train_reg.shape[1:]) == [1, n_assets],
        "y_cls_shape_ok": list(bundle.y_train_cls.shape[1:]) == [1, n_assets, 2],
        "sector_feature_dims_sum_to_assets_times_features": sum(x.shape[-1] for x in bundle.x_train_list)
        == n_assets * int(bundle.meta["features_per_asset"]),
        "all_split_lengths_match": (
            all(x.shape[0] == bundle.y_train_reg.shape[0] for x in bundle.x_train_list)
            and all(x.shape[0] == bundle.y_val_reg.shape[0] for x in bundle.x_val_list)
            and all(x.shape[0] == bundle.y_test_reg.shape[0] for x in bundle.x_test_list)
        ),
        "cls_one_hot_sum_is_one": bool(torch.allclose(bundle.y_train_cls.sum(dim=-1), torch.ones_like(bundle.y_train_reg))),
    }

    return {
        "meta": bundle.meta,
        "checks": checks,
        "arrays": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check proposed Torch data loading and sector split.")
    parser.add_argument("--root", default=".", help="Path to D:/CSCNet/code_v2/proposed equivalent.")
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--use_signature", action="store_true")
    parser.add_argument("--signature_level", type=int, default=2)
    parser.add_argument(
        "--out",
        default="outputs/torch_proposed/data_check.json",
        help="Output JSON path relative to root unless absolute.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    bundle = load_proposed_djia_data(
        root=root,
        train_val_split=args.train_val_split,
        use_signature=args.use_signature,
        signature_level=args.signature_level,
    )
    report = build_data_check(bundle)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["checks"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
