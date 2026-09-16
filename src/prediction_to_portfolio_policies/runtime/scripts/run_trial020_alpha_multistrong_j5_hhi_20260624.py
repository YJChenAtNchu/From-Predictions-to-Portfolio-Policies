from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_data_formal_split as formal_data
import torch_train_profitboost_formal_val as core
from scripts.evaluate_independent_multistrong_alpha_fixedj5_20260608 import (
    alpha_weighted_ensemble,
    best_checkpoint_alpha,
    clean_metrics,
    load_prediction,
)
from scripts.run_paper_sweep_f2_f3_fs_solver_hhi_20260619 import (
    CONDITIONS,
    DATASETS,
    SPARSE_INIT,
    SPARSE_INIT_FAN_IN,
    SPARSE_INIT_SCALE,
    evaluate_final_test,
    load_bundle,
    make_builder,
    patch_alt_split_date_checks,
)
from torch_multiweak_ensemble import evaluate_policy_ensemble


METHOD = "trial020_alpha_multistrong_j5_hhi"
FIXED_J = 5
OUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "torch_proposed"
    / "trial020_alpha_multistrong_j5_hhi_20260624"
)
REPORT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "daily_report"
    / "trial020_alpha_multistrong_j5_hhi_2026-06-24"
)

# Architecture and hyperparameters are fixed from the minimax-rank HPO result.
SELECTED_CONDITIONS = [
    "F3_a2_5day__lista__hhi",
]
DEFAULT_DATASETS = ["DJIA30_2026", "SP500_TOP50_EX_PLTR_2026"]

TRIAL020_CONFIG: dict[str, Any] = {
    "lr": 2e-5,
    "dropout_rate": 0.2,
    "weight_decay": 1e-4,
    "beta": 0.8,
    "lista_unroll_steps": 6,
    "eta_k": 0.5,
    "eta_omega": 2.0,
    "rank_weight": 1.0,
    "use_ema_loss_norm": True,
    "loss_mode": "ce_mse_rank",
    "decision_rule": "hhi_effective",
    "k_min": 1,
    "k_max": 5,
    "fee": 0.001,
    "trading_freq": 5.0,
    "batch_size": 16,
    "ema_alpha": 0.1,
    "weight_update_mode": "sample_specific",
    "profit_source": "one_step_sum",
    "inner_train_val_split": 0.9,
    "lbp_unroll_steps": 4,
}


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
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def learner_seed(experiment_seed: int, learner_index: int) -> int:
    return int(experiment_seed) * 1000 + int(learner_index) + 1


def learner_dir(dataset: str, condition_name: str, experiment_seed: int, index: int) -> Path:
    current_seed = learner_seed(experiment_seed, index)
    return (
        OUT_ROOT
        / dataset
        / condition_name
        / f"experiment_seed{experiment_seed:03d}"
        / f"learner{index + 1:02d}_seed{current_seed:05d}"
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def annotate_learner_summary(
    *,
    dataset: str,
    condition_name: str,
    out_dir: Path,
) -> None:
    condition = CONDITIONS[condition_name]
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["deterministic_requested"] = True
    summary["config"]["architecture_variant"] = condition_name
    summary["data_protocol"] = {
        "outer_train": "2009-2020; split by target_end_date",
        "outer_validation": "2021-2023",
        "final_test": "2024-2025",
        "split_by": "target_end_date",
    }
    summary["deterministic_seed_controls"] = [
        "PYTHONHASHSEED",
        "random.seed",
        "numpy.random.seed",
        "torch.manual_seed",
        "torch.cuda.manual_seed_all",
        "cudnn.benchmark=False",
        "cudnn.deterministic=True",
        "torch.use_deterministic_algorithms(True, warn_only=True)",
        "flash/mem-efficient SDP disabled",
    ]
    summary["multistrong_condition_config"] = {
        "method": METHOD,
        "dataset": dataset,
        "condition": condition.name,
        "frontend_family": condition.frontend_family,
        "solver_type": condition.solver_type,
        "display_name": condition.display_name,
        "fixed_J": FIXED_J,
        "decision_rule": "hhi_effective",
        "hyperparameter_source": "trial020 selected by minimize max(DJIA ARR rank, SP50 ARR rank)",
        "trial020_config": TRIAL020_CONFIG,
        "aggregation": "sum(max(alpha_j,0) * policy_j) / sum(max(alpha_j,0))",
        "alpha_source": "training-derived alpha from each learner summary; validation/test are evaluation only",
        "sparse_init": SPARSE_INIT,
        "sparse_init_scale": SPARSE_INIT_SCALE,
        "sparse_init_fan_in": SPARSE_INIT_FAN_IN,
        "mainline_modified": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def train_learner(
    *,
    dataset: str,
    data_root: Path,
    condition_name: str,
    experiment_seed: int,
    index: int,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    condition = CONDITIONS[condition_name]
    out_dir = learner_dir(dataset, condition_name, experiment_seed, index)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["config"].get("decision_rule") != "hhi_effective":
            raise ValueError(f"Wrong decision rule in {summary_path}")
        return summary

    current_seed = learner_seed(experiment_seed, index)
    print(
        f"[Rank1-3 Multi-Strong] dataset={dataset} condition={condition_name} "
        f"experiment_seed={experiment_seed:03d} learner={index + 1}/{FIXED_J} "
        f"learner_seed={current_seed}",
        flush=True,
    )
    set_global_seed(current_seed, deterministic=True)
    core.build_model_from_data = make_builder(condition)
    report = core.train_profitboost_formal_val(
        data_root=data_root,
        out_dir=out_dir,
        seed=current_seed,
        max_outer_rounds=1 if smoke else 50,
        inner_epochs=2 if smoke else 1000,
        inner_patience=0 if smoke else 5,
        outer_patience=1 if smoke else 5,
        batch_size=int(TRIAL020_CONFIG["batch_size"]),
        lr=float(TRIAL020_CONFIG["lr"]),
        weight_decay=float(TRIAL020_CONFIG["weight_decay"]),
        beta=float(TRIAL020_CONFIG["beta"]),
        loss_mode=str(TRIAL020_CONFIG["loss_mode"]),
        rank_weight=float(TRIAL020_CONFIG["rank_weight"]),
        use_ema_loss_norm=bool(TRIAL020_CONFIG["use_ema_loss_norm"]),
        ema_alpha=float(TRIAL020_CONFIG["ema_alpha"]),
        fee=float(TRIAL020_CONFIG["fee"]),
        k_min=int(TRIAL020_CONFIG["k_min"]),
        k_max=int(TRIAL020_CONFIG["k_max"]),
        decision_rule=str(TRIAL020_CONFIG["decision_rule"]),
        eta_k=float(TRIAL020_CONFIG["eta_k"]),
        eta_omega=float(TRIAL020_CONFIG["eta_omega"]),
        trading_freq=float(TRIAL020_CONFIG["trading_freq"]),
        frontend_type=condition.solver_type,
        lista_unroll_steps=int(TRIAL020_CONFIG["lista_unroll_steps"]),
        lbp_unroll_steps=int(TRIAL020_CONFIG["lbp_unroll_steps"]),
        dropout_rate=float(TRIAL020_CONFIG["dropout_rate"]),
        use_signature=condition.use_signature,
        signature_level=condition.signature_level or 2,
        weight_update_mode=str(TRIAL020_CONFIG["weight_update_mode"]),
        profit_source=str(TRIAL020_CONFIG["profit_source"]),
        inner_train_val_split=float(TRIAL020_CONFIG["inner_train_val_split"]),
        device=device,
    )
    annotate_learner_summary(dataset=dataset, condition_name=condition_name, out_dir=out_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def ensure_final_prediction(
    *,
    dataset: str,
    data_root: Path,
    condition_name: str,
    directory: Path,
    device: str,
) -> dict[str, np.ndarray]:
    paths = {
        "policy": directory / "final_test_weights.npy",
        "cls_prob": directory / "final_test_cls_out.npy",
        "reg_out": directory / "final_test_reg_out.npy",
    }
    if all(path.exists() for path in paths.values()):
        return {key: np.load(path).astype(np.float32) for key, path in paths.items()}
    evaluate_final_test(
        condition=CONDITIONS[condition_name],
        dataset=dataset,
        data_root=data_root,
        out_dir=directory,
        device=device,
    )
    return {key: np.load(path).astype(np.float32) for key, path in paths.items()}


def metric_row(
    *,
    dataset: str,
    condition_name: str,
    experiment_seed: int,
    split: str,
    metrics: dict[str, Any],
    alphas: list[float],
    alpha_rounds: list[int],
    fallback: bool,
) -> dict[str, Any]:
    condition = CONDITIONS[condition_name]
    financial = metrics["financial_metrics"]
    alpha_array = np.asarray(alphas, dtype=np.float64)
    normalized = (
        alpha_array / alpha_array.sum()
        if alpha_array.sum() > 1e-12
        else np.ones(FIXED_J, dtype=np.float64) / FIXED_J
    )
    return {
        "dataset": dataset,
        "condition": condition_name,
        "method": condition.display_name,
        "frontend_family": condition.frontend_family,
        "solver_type": condition.solver_type,
        "experiment_seed": experiment_seed,
        "split": split,
        "fixed_J": FIXED_J,
        "ACC": metrics["acc"],
        "APV": financial["APV"],
        "ARR": financial["APR"],
        "AVol": financial["AVol"],
        "ASR": financial["ASR"],
        "MDD": financial["MDD"],
        "CR": financial["CR"],
        "Turnover": metrics["total_turnover"],
        "alpha_total": float(alpha_array.sum()),
        "alpha_min": float(alpha_array.min()),
        "alpha_mean": float(alpha_array.mean()),
        "alpha_max": float(alpha_array.max()),
        "normalized_alpha_max": float(normalized.max()),
        "effective_learner_number": float(1.0 / np.sum(normalized**2)),
        "equal_weight_fallback": bool(fallback),
        "alphas": ",".join(f"{value:.10f}" for value in alphas),
        "alpha_outer_rounds": ",".join(str(value) for value in alpha_rounds),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["condition"], row["split"]) for row in rows})
    for dataset, condition_name, split in keys:
        group = [
            row
            for row in rows
            if row["dataset"] == dataset and row["condition"] == condition_name and row["split"] == split
        ]
        first = group[0]
        item: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition_name,
            "method": first["method"],
            "frontend_family": first["frontend_family"],
            "solver_type": first["solver_type"],
            "split": split,
            "n_seeds": len(group),
            "positive_seeds": sum(float(row["APV"]) > 1.0 for row in group),
            "fixed_J": FIXED_J,
            "fallback_seeds": sum(bool(row["equal_weight_fallback"]) for row in group),
        }
        for metric in [
            "ACC",
            "APV",
            "ARR",
            "AVol",
            "ASR",
            "MDD",
            "CR",
            "Turnover",
            "alpha_total",
            "normalized_alpha_max",
            "effective_learner_number",
        ]:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
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


def merge_write_csv(path: Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    keyed = {
        (str(row["dataset"]), str(row["condition"]), int(row["experiment_seed"]), str(row["split"])): row
        for row in existing_rows
    }
    for row in new_rows:
        keyed[(row["dataset"], row["condition"], int(row["experiment_seed"]), row["split"])] = row
    merged = list(keyed.values())
    merged.sort(key=lambda row: (row["dataset"], row["condition"], int(row["experiment_seed"]), row["split"]))
    write_csv(path, merged)
    return merged


def run_seed(
    *,
    dataset: str,
    data_root: Path,
    condition_name: str,
    experiment_seed: int,
    device: str,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition = CONDITIONS[condition_name]
    bundle = load_bundle(condition, data_root, device)
    outer_predictions = []
    final_predictions = []
    alphas = []
    alpha_rounds = []

    for index in range(FIXED_J):
        summary = train_learner(
            dataset=dataset,
            data_root=data_root,
            condition_name=condition_name,
            experiment_seed=experiment_seed,
            index=index,
            device=device,
            smoke=smoke,
        )
        directory = learner_dir(dataset, condition_name, experiment_seed, index)
        alpha, alpha_round = best_checkpoint_alpha(summary)
        alphas.append(alpha)
        alpha_rounds.append(alpha_round)
        outer_predictions.append(load_prediction(directory, "outer_val"))
        final_predictions.append(
            ensure_final_prediction(
                dataset=dataset,
                data_root=data_root,
                condition_name=condition_name,
                directory=directory,
                device=device,
            )
        )

    outer_ensemble, outer_fallback = alpha_weighted_ensemble(outer_predictions, alphas, device)
    final_ensemble, final_fallback = alpha_weighted_ensemble(final_predictions, alphas, device)
    outer_metrics = evaluate_policy_ensemble(
        outer_ensemble,
        bundle.y_outer_val_reg,
        bundle.y_outer_val_cls,
        fee=0.001,
        trading_freq=5.0,
    )
    final_metrics = evaluate_policy_ensemble(
        final_ensemble,
        bundle.y_final_test_reg,
        bundle.y_final_test_cls,
        fee=0.001,
        trading_freq=5.0,
    )
    rows = [
        metric_row(
            dataset=dataset,
            condition_name=condition_name,
            experiment_seed=experiment_seed,
            split="outer_validation",
            metrics=outer_metrics,
            alphas=alphas,
            alpha_rounds=alpha_rounds,
            fallback=outer_fallback,
        ),
        metric_row(
            dataset=dataset,
            condition_name=condition_name,
            experiment_seed=experiment_seed,
            split="final_test",
            metrics=final_metrics,
            alphas=alphas,
            alpha_rounds=alpha_rounds,
            fallback=final_fallback,
        ),
    ]
    detail = {
        "dataset": dataset,
        "condition": condition_name,
        "experiment_seed": experiment_seed,
        "decision_rule": "hhi_effective",
        "J": FIXED_J,
        "alphas": alphas,
        "alpha_outer_rounds": alpha_rounds,
        "outer_validation": clean_metrics(outer_metrics),
        "final_test": clean_metrics(final_metrics),
    }
    return rows, detail


def parse_list(value: str, default: list[str]) -> list[str]:
    if value == "default":
        return list(default)
    if value == "all":
        return list(DATASETS.keys()) if default == DEFAULT_DATASETS else list(CONDITIONS.keys())
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="default")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    patch_alt_split_date_checks()
    core.load_proposed_djia_formal_data = formal_data.load_proposed_djia_formal_data
    core.set_seed = set_global_seed
    device = resolve_device(args.device)

    selected_conditions = parse_list(args.condition, SELECTED_CONDITIONS)
    selected_datasets = parse_list(args.dataset, DEFAULT_DATASETS)
    invalid_conditions = [name for name in selected_conditions if name not in CONDITIONS]
    invalid_datasets = [name for name in selected_datasets if name not in DATASETS]
    if invalid_conditions or invalid_datasets:
        raise ValueError(f"Invalid conditions={invalid_conditions}, datasets={invalid_datasets}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_new_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for dataset in selected_datasets:
        data_root = DATASETS[dataset]
        details[dataset] = {}
        for condition_name in selected_conditions:
            details[dataset][condition_name] = {}
            for seed in range(args.seed_start, args.seed_end + 1):
                rows, detail = run_seed(
                    dataset=dataset,
                    data_root=data_root,
                    condition_name=condition_name,
                    experiment_seed=seed,
                    device=device,
                    smoke=args.smoke,
                )
                all_new_rows.extend(rows)
                details[dataset][condition_name][f"seed{seed:03d}"] = detail
                print(f"[complete] {dataset} {condition_name} experiment_seed{seed:03d}", flush=True)

    prefix = "smoke_" if args.smoke else ""
    per_seed_path = REPORT_ROOT / f"{prefix}trial020_alpha_multistrong_j5_per_seed.csv"
    merged = merge_write_csv(per_seed_path, all_new_rows)
    summaries = summarize(merged)
    write_csv(REPORT_ROOT / f"{prefix}trial020_alpha_multistrong_j5_group_summary.csv", summaries)
    payload = {
        "method": METHOD,
        "rank_basis": "trial020 selected by minimax rank over DJIA/SP50 validation ARR",
        "selected_conditions": SELECTED_CONDITIONS,
        "datasets": selected_datasets,
        "trial020_config": TRIAL020_CONFIG,
        "decision_rule": "hhi_effective",
        "J": FIXED_J,
        "seed_range": [args.seed_start, args.seed_end],
        "sparse_init": SPARSE_INIT,
        "sparse_init_scale": SPARSE_INIT_SCALE,
        "sparse_init_fan_in": SPARSE_INIT_FAN_IN,
        "deterministic_requested": True,
        "output_root": str(OUT_ROOT),
        "report_root": str(REPORT_ROOT),
        "summary": summaries,
        "new_details": details,
        "mainline_modified": False,
    }
    (REPORT_ROOT / f"{prefix}summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
