from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_data_formal_split as formal_data
import torch_train_profitboost_formal_val as core
from torch_model_paper_sweep_frontend_solver_20260619 import (
    FRONTEND_SPECS,
    SPARSE_INIT,
    SPARSE_INIT_FAN_IN,
    SPARSE_INIT_SCALE,
    build_paper_sweep_model_from_data,
)
from torch_train_smoke import _take_batch


OUT_ROOT = PROJECT_ROOT / "outputs" / "torch_proposed" / "paper_sweep_f2_f3_fs_solver_hhi_20260619"
REPORT_ROOT = PROJECT_ROOT / "outputs" / "daily_report" / "paper_sweep_f2_f3_fs_solver_hhi_2026-06-19"
DATASETS = {
    "DJIA30_2026": PROJECT_ROOT / "data" / "djia30_2026_seq15_tar5_stride5_ratio_val2021_2023_test2024_2025",
    "SP500_TOP50_EX_PLTR_2026": PROJECT_ROOT / "data" / "sp500_top50_ex_pltr_2026_seq15_tar5_stride5_ratio_val2021_2023_test2024_2025",
}
FRONTENDS = {
    "F2_temporal_independent": {
        "display_name": "F2 Temporal-independent Asset-aligned Sparse CSCNet",
        "use_signature": False,
        "signature_level": None,
        "backend": "sector GRU + sector attention + heads",
        "neural_predictor": "CSC -> sector GRU -> sequence/sector attention -> regression/classification heads",
    },
    "F3_a2_5day": {
        "display_name": "F3 A2 5-day Asset-aligned Sparse CSCNet",
        "use_signature": False,
        "signature_level": None,
        "backend": "sector GRU + sector attention + heads",
        "neural_predictor": "CSC -> sector GRU -> sequence/sector attention -> regression/classification heads",
    },
    "FS_sl_hsa_l2": {
        "display_name": "FS SL-HSA L2 Signature Sparse CSCNet",
        "use_signature": True,
        "signature_level": 2,
        "backend": "intra-sector asset attention + mean sector pooling + inter-sector attention + asset-wise heads",
        "neural_predictor": "asset-wise Signature L2 -> CSC -> intra/inter-sector attention -> regression/classification heads",
    },
}
SOLVERS = ("ista", "lista", "fista", "lbp")


@dataclass(frozen=True)
class Condition:
    name: str
    frontend_family: str
    solver_type: str
    display_name: str
    use_signature: bool
    signature_level: int | None
    backend: str
    neural_predictor: str


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            # Prefer deterministic scaled-dot-product attention kernels for
            # MultiheadAttention used by the sector/signature predictors.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def patch_alt_split_date_checks() -> None:
    def _assert_target_date_rules(
        outer_train_dates: pd.DataFrame,
        outer_val_dates: pd.DataFrame,
        final_test_dates: pd.DataFrame,
    ) -> dict[str, bool]:
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
            raise ValueError(f"Alt split target-date rule failed: {checks}")
        return checks

    formal_data._assert_target_date_rules = _assert_target_date_rules
    core.load_proposed_djia_formal_data = formal_data.load_proposed_djia_formal_data
    core.set_seed = set_global_seed


def ensure_sp50_alt_split() -> None:
    target = DATASETS["SP500_TOP50_EX_PLTR_2026"]
    if (target / "dataset_summary.json").is_file():
        return
    subprocess.run(
        [sys.executable, "scripts/build_sp500_alt_split_val2021_2023_test2024_2025_20260619.py"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def build_conditions() -> dict[str, Condition]:
    conditions: dict[str, Condition] = {}
    for frontend_family, meta in FRONTENDS.items():
        for solver in SOLVERS:
            name = f"{frontend_family}__{solver}__hhi"
            conditions[name] = Condition(
                name=name,
                frontend_family=frontend_family,
                solver_type=solver,
                display_name=f"{meta['display_name']} + {solver.upper()} + HHI",
                use_signature=bool(meta["use_signature"]),
                signature_level=meta["signature_level"],
                backend=str(meta["backend"]),
                neural_predictor=str(meta["neural_predictor"]),
            )
    return conditions


CONDITIONS = build_conditions()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_builder(condition: Condition):
    def builder(bundle, **kwargs):
        return build_paper_sweep_model_from_data(
            bundle,
            frontend_family=condition.frontend_family,
            solver_type=condition.solver_type,
            **kwargs,
        )

    return builder


def load_bundle(condition: Condition, data_root: Path, device: str, config: dict[str, Any] | None = None):
    return formal_data.load_proposed_djia_formal_data(
        root=data_root,
        inner_train_val_split=float((config or {}).get("inner_train_val_split", 0.9)),
        device=device,
        use_signature=condition.use_signature,
        signature_level=condition.signature_level or 2,
    )


def evaluate_final_test(
    *,
    condition: Condition,
    dataset: str,
    data_root: Path,
    out_dir: Path,
    device: str,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    seed = int(config["seed"])
    resolved_device = resolve_device(device)
    set_global_seed(seed)

    bundle = load_bundle(condition, data_root, resolved_device, config)
    model = make_builder(condition)(
        bundle,
        frontend_type=str(config.get("frontend_type", condition.solver_type)),
        lista_unroll_steps=int(config.get("lista_unroll_steps", 4)),
        lbp_unroll_steps=int(config.get("lbp_unroll_steps", 4)),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
    ).to(resolved_device)
    warmup = torch.arange(
        min(int(config.get("batch_size", 16)), bundle.y_train_reg.shape[0]),
        device=resolved_device,
    )
    model(_take_batch(bundle.x_train_list, warmup))
    model.load_state_dict(torch.load(out_dir / f"model_seed{seed:03d}.pt", map_location=resolved_device))
    model.eval()

    final_metrics = core.evaluate_split(
        model,
        bundle.x_final_test_list,
        bundle.y_final_test_reg,
        bundle.y_final_test_cls,
        fee=float(config.get("fee", 0.001)),
        k_max=int(config.get("k_max", 5)),
        k_min=int(config.get("k_min", 1)),
        trading_freq=float(config.get("trading_freq", 5.0)),
        decision_rule=str(config.get("decision_rule", "hhi_effective")),
        eta_k=float(config.get("eta_k", 1.0)),
        eta_omega=float(config.get("eta_omega", 1.0)),
        return_outputs=True,
    )
    outputs = final_metrics.pop("_outputs")
    weights = final_metrics.pop("_weights")
    selected_k = final_metrics.pop("_selected_k")
    diagnostics = final_metrics.pop("_decision_diagnostics")
    wealth = final_metrics.pop("_wealth")
    paths = {
        "reg_out": out_dir / "final_test_reg_out.npy",
        "cls_out": out_dir / "final_test_cls_out.npy",
        "weights": out_dir / "final_test_weights.npy",
        "selected_k": out_dir / "final_test_selected_k.npy",
        "wealth_curve": out_dir / "final_test_wealth_curve.npy",
        "hhi": out_dir / "final_test_hhi.npy",
        "n_eff": out_dir / "final_test_n_eff.npy",
    }
    np.save(paths["reg_out"], outputs["reg_out"].detach().cpu().numpy())
    np.save(paths["cls_out"], outputs["cls_out"].detach().cpu().numpy())
    np.save(paths["weights"], weights.detach().cpu().numpy())
    np.save(paths["selected_k"], selected_k.detach().cpu().numpy())
    np.save(paths["wealth_curve"], wealth.detach().cpu().numpy())
    hhi = diagnostics.get("hhi")
    n_eff = diagnostics.get("n_eff")
    if isinstance(hhi, torch.Tensor):
        np.save(paths["hhi"], hhi.detach().cpu().numpy())
    if isinstance(n_eff, torch.Tensor):
        np.save(paths["n_eff"], n_eff.detach().cpu().numpy())

    spec = FRONTEND_SPECS[condition.frontend_family]
    summary["config"]["final_test_evaluated"] = True
    summary["config"]["architecture_variant"] = condition.name
    summary["config"]["sparse_init"] = SPARSE_INIT
    summary["config"]["sparse_init_scale"] = SPARSE_INIT_SCALE
    summary["config"]["sparse_init_fan_in"] = SPARSE_INIT_FAN_IN
    summary["data_protocol"] = {
        "outer_train": "2009-2020; split by target_end_date",
        "outer_validation": "2021-2023; configuration assessment",
        "final_test": "2024-2025; post-training evaluation",
        "split_by": "target_end_date",
    }
    summary["metrics"]["final_test"] = final_metrics
    summary["final_test"] = final_metrics
    summary.setdefault("outputs", {}).setdefault("splits", {})["final_test"] = {
        key: str(path) if path.is_file() else None for key, path in paths.items()
    }
    summary["paper_sweep_config"] = {
        "dataset": dataset,
        "condition": condition.name,
        "frontend_family": condition.frontend_family,
        "solver_type": condition.solver_type,
        "decision_rule": "hhi_effective",
        "sparse_init": SPARSE_INIT,
        "sparse_init_scale": SPARSE_INIT_SCALE,
        "sparse_init_fan_in": SPARSE_INIT_FAN_IN,
        "backend": condition.backend,
        "neural_predictor": condition.neural_predictor,
        "first_layer_kernel": list(spec.first_kernel),
        "first_layer_stride": list(spec.first_stride),
        "layer2_kernel": list(spec.layer2_kernel),
        "layer2_stride": list(spec.layer2_stride),
        "layer3_kernel": list(spec.layer3_kernel),
        "layer3_stride": list(spec.layer3_stride),
        "data_root": str(data_root),
        "mainline_modified": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_metrics


def metric_row(
    *,
    dataset: str,
    condition: Condition,
    seed: int,
    split: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    fin = metrics["financial_metrics"]
    spec = FRONTEND_SPECS[condition.frontend_family]
    return {
        "dataset": dataset,
        "condition": condition.name,
        "frontend_family": condition.frontend_family,
        "solver_type": condition.solver_type,
        "method": condition.display_name,
        "backend": condition.backend,
        "seed": seed,
        "split": split,
        "first_kernel": str(tuple(spec.first_kernel)),
        "first_stride": str(tuple(spec.first_stride)),
        "layer2_kernel": str(tuple(spec.layer2_kernel)),
        "layer2_stride": str(tuple(spec.layer2_stride)),
        "layer3_kernel": str(tuple(spec.layer3_kernel)),
        "layer3_stride": str(tuple(spec.layer3_stride)),
        "ACC": metrics["acc"],
        "APV": fin["APV"],
        "ARR": fin["APR"],
        "AVol": fin["AVol"],
        "ASR": fin["ASR"],
        "MDD": fin["MDD"],
        "CR": fin["CR"],
        "Turnover": metrics["total_turnover"],
        "mean_selected_k": metrics.get("mean_selected_k"),
        "std_selected_k": metrics.get("std_selected_k"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["dataset"], r["condition"], r["split"]) for r in rows})
    for dataset, condition_name, split in keys:
        group = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition_name and r["split"] == split]
        first = group[0]
        item: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition_name,
            "frontend_family": first["frontend_family"],
            "solver_type": first["solver_type"],
            "method": first["method"],
            "backend": first["backend"],
            "split": split,
            "n_seeds": len(group),
            "positive_seeds": sum(float(r["APV"]) > 1.0 for r in group),
            "first_kernel": first["first_kernel"],
            "first_stride": first["first_stride"],
            "layer2_kernel": first["layer2_kernel"],
            "layer2_stride": first["layer2_stride"],
            "layer3_kernel": first["layer3_kernel"],
            "layer3_stride": first["layer3_stride"],
        }
        for metric in ["ACC", "APV", "ARR", "AVol", "ASR", "MDD", "CR", "Turnover", "mean_selected_k", "std_selected_k"]:
            values = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(condition: Condition, dataset: str, data_root: Path, seed: int, device: str, smoke: bool) -> list[dict[str, Any]]:
    core.build_model_from_data = make_builder(condition)
    set_global_seed(seed)
    out_dir = OUT_ROOT / condition.name / dataset / f"seed{seed:03d}"
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        print(f"[paper-sweep] {dataset} {condition.name} seed{seed:03d}", flush=True)
        core.train_profitboost_formal_val(
            data_root=data_root,
            out_dir=out_dir,
            seed=seed,
            max_outer_rounds=1 if smoke else 50,
            inner_epochs=2 if smoke else 1000,
            inner_patience=0 if smoke else 5,
            outer_patience=1 if smoke else 5,
            batch_size=16,
            lr=1e-5,
            weight_decay=0.0,
            beta=0.6,
            loss_mode="ce_mse_rank",
            rank_weight=1.0,
            use_ema_loss_norm=True,
            ema_alpha=0.1,
            fee=0.001,
            k_min=1,
            k_max=5,
            decision_rule="hhi_effective",
            eta_k=1.0,
            eta_omega=1.0,
            trading_freq=5.0,
            frontend_type=condition.solver_type,
            lista_unroll_steps=4,
            lbp_unroll_steps=4,
            dropout_rate=0.1,
            use_signature=condition.use_signature,
            signature_level=condition.signature_level or 2,
            weight_update_mode="sample_specific",
            profit_source="one_step_sum",
            inner_train_val_split=0.9,
            device=device,
        )
    else:
        print(f"[skip train] {dataset} {condition.name} seed{seed:03d}", flush=True)
    final_metrics = evaluate_final_test(
        condition=condition,
        dataset=dataset,
        data_root=data_root,
        out_dir=out_dir,
        device=device,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return [
        metric_row(dataset=dataset, condition=condition, seed=seed, split="outer_validation", metrics=summary["outer_validation"]),
        metric_row(dataset=dataset, condition=condition, seed=seed, split="final_test", metrics=final_metrics),
    ]


def selected_conditions(selector: str) -> list[Condition]:
    if selector == "all":
        return list(CONDITIONS.values())
    return [CONDITIONS[selector]]


def selected_datasets(selector: str) -> dict[str, Path]:
    if selector == "all":
        return DATASETS
    return {selector: DATASETS[selector]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["all", *CONDITIONS.keys()], default="all")
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    patch_alt_split_date_checks()
    ensure_sp50_alt_split()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for dataset, data_root in selected_datasets(args.dataset).items():
        for condition in selected_conditions(args.condition):
            for seed in range(args.seed_start, args.seed_end + 1):
                all_rows.extend(run_one(condition, dataset, data_root, seed, args.device, args.smoke))

    prefix = "smoke_" if args.smoke else ""
    summary_rows = summarize(all_rows)
    write_csv(REPORT_ROOT / f"{prefix}paper_sweep_per_seed.csv", all_rows)
    write_csv(REPORT_ROOT / f"{prefix}paper_sweep_group_summary.csv", summary_rows)
    manifest = {
        "experiment": "F2/F3/FS x ISTA/LISTA/FISTA/LBP x HHI",
        "output_root": str(OUT_ROOT),
        "report_root": str(REPORT_ROOT),
        "split": {
            "DJIA30_2026": "train 2009-2020, validation 2021-2023, test 2024-2025",
            "SP500_TOP50_EX_PLTR_2026": "train 2009-2020, validation 2021-2023 retained, final report focuses on test 2024-2025",
            "split_by": "target_end_date",
        },
        "seed_range": [args.seed_start, args.seed_end],
        "decision_rule": "hhi_effective",
        "sparse_init": SPARSE_INIT,
        "sparse_init_scale": SPARSE_INIT_SCALE,
        "sparse_init_fan_in": SPARSE_INIT_FAN_IN,
        "deterministic_requested": True,
        "seed_controls": [
            "PYTHONHASHSEED",
            "random.seed",
            "numpy.random.seed",
            "torch.manual_seed",
            "torch.cuda.manual_seed_all",
            "cudnn.benchmark=False",
            "cudnn.deterministic=True",
            "torch.use_deterministic_algorithms(True, warn_only=True)",
        ],
        "frontends": {
            name: {
                **meta,
                "kernel_spec": {
                    "first_kernel": list(FRONTEND_SPECS[name].first_kernel),
                    "first_stride": list(FRONTEND_SPECS[name].first_stride),
                    "layer2_kernel": list(FRONTEND_SPECS[name].layer2_kernel),
                    "layer2_stride": list(FRONTEND_SPECS[name].layer2_stride),
                    "layer3_kernel": list(FRONTEND_SPECS[name].layer3_kernel),
                    "layer3_stride": list(FRONTEND_SPECS[name].layer3_stride),
                },
            }
            for name, meta in FRONTENDS.items()
        },
        "solvers": list(SOLVERS),
        "summary": summary_rows,
        "mainline_modified": False,
    }
    (REPORT_ROOT / f"{prefix}summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
