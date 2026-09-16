from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_data_execution_t1_t6 as execution_data  # noqa: E402
import torch_train_profitboost_formal_val as core  # noqa: E402
from scripts import run_trial020_alpha_multistrong_j5_hhi_20260624 as reference  # noqa: E402
from torch_model_nosparse_gru_attention_20260705 import (  # noqa: E402
    build_nosparse_gru_attention_model_from_data,
)
from torch_multiweak_ensemble import evaluate_policy_ensemble  # noqa: E402
from torch_train_smoke import _take_batch  # noqa: E402


METHOD = "execution_t1_t6_trial020_proposed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "execution_t1_t6_20260902" / "proposed"
REPORT_ROOT = PROJECT_ROOT / "outputs" / "daily_report" / "execution_t1_t6_2026-09-02"
DATA_ROOT = PROJECT_ROOT / "data" / "execution_t1_t6_20260902"
FULL_CONDITION_NAME = "F3_a2_5day__lista__hhi"
DATASETS = {
    "DJIA30_2026": DATA_ROOT / "DJIA30_2026",
    "SP500_TOP50_EX_PLTR_2026": DATA_ROOT / "SP500_TOP50_EX_PLTR_2026",
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED": DATA_ROOT
    / "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED",
}
CONFIG = dict(reference.TRIAL020_CONFIG)


def condition_for(variant: str) -> Any:
    if variant in {"full_hhi", "fixedk5"}:
        return reference.CONDITIONS[FULL_CONDITION_NAME]
    if variant == "nosparse_hhi":
        return SimpleNamespace(
            name="M4_nosparse_gru_attention__none__hhi",
            frontend_family="M4_nosparse_gru_attention",
            solver_type="none",
            display_name="No Sparse + sector GRU/Attention",
            use_signature=False,
            signature_level=None,
            backend="sector GRU + attention + heads",
            neural_predictor="raw normalized OHLC -> sector GRU -> attention -> heads",
        )
    raise ValueError(f"Unknown variant={variant}")


def config_for(variant: str) -> dict[str, Any]:
    config = dict(CONFIG)
    if variant == "fixedk5":
        config.update({"decision_rule": "fixed", "k_min": 1, "k_max": 5})
    return config


def builder_for(variant: str, condition: Any) -> Callable[..., torch.nn.Module]:
    if variant == "nosparse_hhi":
        def build_nosparse(bundle: Any, **kwargs: Any) -> torch.nn.Module:
            return build_nosparse_gru_attention_model_from_data(bundle, **kwargs)

        return build_nosparse
    return reference.make_builder(condition)


def learner_seed(experiment_seed: int, learner_index: int) -> int:
    return int(experiment_seed) * 1000 + int(learner_index) + 1


def learner_dir(variant: str, dataset: str, experiment_seed: int, learner_index: int, smoke: bool) -> Path:
    root = OUTPUT_ROOT / ("smoke" if smoke else "formal_bank") / variant
    seed = learner_seed(experiment_seed, learner_index)
    return (
        root
        / dataset
        / f"experiment_seed{experiment_seed:03d}"
        / f"learner{learner_index + 1:02d}_seed{seed:05d}"
    )


def _load_bundle(
    data_root: Path,
    condition: Any,
    device: str,
    *,
    allow_final_test: bool,
    config: dict[str, Any],
) -> Any:
    return execution_data.load_proposed_execution_data(
        root=data_root,
        inner_train_val_split=float(config["inner_train_val_split"]),
        device=device,
        use_signature=condition.use_signature,
        signature_level=condition.signature_level or 2,
        allow_final_test=allow_final_test,
    )


def train_learner(
    *,
    variant: str,
    dataset: str,
    data_root: Path,
    experiment_seed: int,
    learner_index: int,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    condition = condition_for(variant)
    config = config_for(variant)
    directory = learner_dir(variant, dataset, experiment_seed, learner_index, smoke)
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = summary.get("execution_t1_t6_audit", {})
        expected = {
            "target_definition": execution_data.EXPECTED_TARGET,
            "variant": variant,
            "smoke": smoke,
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Refusing incompatible cached run: {summary_path}")
        return summary

    seed = learner_seed(experiment_seed, learner_index)
    reference.set_global_seed(seed, deterministic=True)
    core.set_seed = reference.set_global_seed
    core.load_proposed_djia_formal_data = lambda **kwargs: execution_data.load_proposed_execution_data(
        **kwargs, allow_final_test=False
    )
    core.build_model_from_data = builder_for(variant, condition)
    report = core.train_profitboost_formal_val(
        data_root=data_root,
        out_dir=directory,
        seed=seed,
        max_outer_rounds=1 if smoke else 50,
        inner_epochs=2 if smoke else 1000,
        inner_patience=0 if smoke else 5,
        outer_patience=1 if smoke else 5,
        batch_size=int(config["batch_size"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        beta=float(config["beta"]),
        loss_mode=str(config["loss_mode"]),
        rank_weight=float(config["rank_weight"]),
        use_ema_loss_norm=bool(config["use_ema_loss_norm"]),
        ema_alpha=float(config["ema_alpha"]),
        fee=float(config["fee"]),
        k_min=int(config["k_min"]),
        k_max=int(config["k_max"]),
        decision_rule=str(config["decision_rule"]),
        eta_k=float(config["eta_k"]),
        eta_omega=float(config["eta_omega"]),
        trading_freq=float(config["trading_freq"]),
        frontend_type=condition.solver_type,
        lista_unroll_steps=int(config["lista_unroll_steps"]),
        lbp_unroll_steps=int(config["lbp_unroll_steps"]),
        dropout_rate=float(config["dropout_rate"]),
        use_signature=condition.use_signature,
        signature_level=condition.signature_level or 2,
        weight_update_mode=str(config["weight_update_mode"]),
        profit_source=str(config["profit_source"]),
        inner_train_val_split=float(config["inner_train_val_split"]),
        device=device,
    )
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["execution_t1_t6_audit"] = {
        "method": METHOD,
        "dataset": dataset,
        "variant": variant,
        "experiment_seed": experiment_seed,
        "learner_index": learner_index + 1,
        "learner_seed": seed,
        "smoke": smoke,
        "deterministic": True,
        "information_set": "adjusted OHLC through close[t]",
        "decision_time": "after close[t]",
        "execution": "close[t+1]",
        "target_definition": execution_data.EXPECTED_TARGET,
        "outer_validation": "2021-2023 complete execution interval",
        "final_test": "sealed_not_loaded",
        "checkpoint_selection": "inner-validation portfolio profit only",
        "config": config,
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def evaluate_final_test(
    *,
    variant: str,
    dataset: str,
    data_root: Path,
    experiment_seed: int,
    learner_index: int,
    device: str,
) -> dict[str, np.ndarray]:
    directory = learner_dir(variant, dataset, experiment_seed, learner_index, False)
    paths = {
        "policy": directory / "final_test_weights.npy",
        "cls_prob": directory / "final_test_cls_out.npy",
        "reg_out": directory / "final_test_reg_out.npy",
    }
    if all(path.is_file() for path in paths.values()):
        return {key: np.load(path).astype(np.float32) for key, path in paths.items()}

    condition = condition_for(variant)
    config = config_for(variant)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = int(summary["config"]["seed"])
    reference.set_global_seed(seed, deterministic=True)
    bundle = _load_bundle(data_root, condition, device, allow_final_test=True, config=config)
    model = builder_for(variant, condition)(
        bundle,
        frontend_type=condition.solver_type,
        lista_unroll_steps=int(config["lista_unroll_steps"]),
        lbp_unroll_steps=int(config["lbp_unroll_steps"]),
        dropout_rate=float(config["dropout_rate"]),
    ).to(device)
    warmup = torch.arange(min(int(config["batch_size"]), bundle.y_train_reg.shape[0]), device=device)
    model(_take_batch(bundle.x_train_list, warmup))
    model.load_state_dict(torch.load(directory / f"model_seed{seed:03d}.pt", map_location=device))
    model.eval()
    metrics = core.evaluate_split(
        model,
        bundle.x_final_test_list,
        bundle.y_final_test_reg,
        bundle.y_final_test_cls,
        fee=float(config["fee"]),
        k_max=int(config["k_max"]),
        k_min=int(config["k_min"]),
        trading_freq=float(config["trading_freq"]),
        decision_rule=str(config["decision_rule"]),
        eta_k=float(config["eta_k"]),
        eta_omega=float(config["eta_omega"]),
        return_outputs=True,
    )
    outputs = metrics.pop("_outputs")
    weights = metrics.pop("_weights")
    selected_k = metrics.pop("_selected_k")
    diagnostics = metrics.pop("_decision_diagnostics")
    wealth = metrics.pop("_wealth")
    np.save(paths["reg_out"], outputs["reg_out"].detach().cpu().numpy())
    np.save(paths["cls_prob"], outputs["cls_out"].detach().cpu().numpy())
    np.save(paths["policy"], weights.detach().cpu().numpy())
    np.save(directory / "final_test_selected_k.npy", selected_k.detach().cpu().numpy())
    np.save(directory / "final_test_wealth_curve.npy", wealth.detach().cpu().numpy())
    for name in ("hhi", "n_eff"):
        value = diagnostics.get(name)
        if isinstance(value, torch.Tensor):
            np.save(directory / f"final_test_{name}.npy", value.detach().cpu().numpy())
    summary["config"]["final_test_evaluated"] = True
    summary["metrics"]["final_test"] = reference.clean_metrics(metrics)
    summary["execution_t1_t6_audit"]["final_test"] = "2024-2025 frozen evaluation completed"
    summary["execution_t1_t6_audit"]["final_test_unsealed_explicitly"] = True
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: np.load(path).astype(np.float32) for key, path in paths.items()}


def metric_row(
    dataset: str,
    variant: str,
    aggregation: str,
    experiment_seed: int,
    split: str,
    metrics: dict[str, Any],
    learner_count: int,
) -> dict[str, Any]:
    financial = metrics["financial_metrics"]
    return {
        "dataset": dataset,
        "variant": variant,
        "aggregation": aggregation,
        "experiment_seed": experiment_seed,
        "split": split,
        "J": learner_count,
        "ACC": metrics["acc"],
        "APV": financial["APV"],
        "ARR": financial["APR"],
        "AVol": financial["AVol"],
        "ASR": financial["ASR"],
        "MDD": financial["MDD"],
        "CR": financial["CR"],
        "Turnover": metrics["total_turnover"],
        "target_definition": execution_data.EXPECTED_TARGET,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["dataset"], row["variant"], row["aggregation"], row["split"]) for row in rows})
    for dataset, variant, aggregation, split in keys:
        group = [
            row for row in rows
            if (row["dataset"], row["variant"], row["aggregation"], row["split"])
            == (dataset, variant, aggregation, split)
        ]
        item: dict[str, Any] = {
            "dataset": dataset,
            "variant": variant,
            "aggregation": aggregation,
            "split": split,
            "n_seeds": len(group),
            "J": group[0]["J"],
        }
        for metric in ("ACC", "APV", "ARR", "AVol", "ASR", "MDD", "CR", "Turnover"):
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    *,
    variant: str,
    dataset: str,
    experiment_seed: int,
    learner_count: int,
    device: str,
    smoke: bool,
    allow_final_test: bool,
) -> list[dict[str, Any]]:
    data_root = DATASETS[dataset]
    condition = condition_for(variant)
    config = config_for(variant)
    summaries = []
    outer_predictions = []
    final_predictions = []
    alphas = []
    for learner_index in range(learner_count):
        summary = train_learner(
            variant=variant,
            dataset=dataset,
            data_root=data_root,
            experiment_seed=experiment_seed,
            learner_index=learner_index,
            device=device,
            smoke=smoke,
        )
        summaries.append(summary)
        directory = learner_dir(variant, dataset, experiment_seed, learner_index, smoke)
        alpha, _ = reference.best_checkpoint_alpha(summary)
        alphas.append(alpha)
        outer_predictions.append(reference.load_prediction(directory, "outer_val"))
        if allow_final_test:
            if smoke:
                raise ValueError("Smoke outputs cannot be used for final test")
            final_predictions.append(
                evaluate_final_test(
                    variant=variant,
                    dataset=dataset,
                    data_root=data_root,
                    experiment_seed=experiment_seed,
                    learner_index=learner_index,
                    device=device,
                )
            )

    validation_bundle = _load_bundle(
        data_root, condition, device, allow_final_test=allow_final_test, config=config
    )
    aggregations: dict[str, tuple[dict[str, torch.Tensor], bool]] = {
        "Alpha-policy": reference.alpha_weighted_ensemble(outer_predictions, alphas, device),
    }
    if learner_count > 1:
        aggregations["Equal-policy"] = reference.alpha_weighted_ensemble(
            outer_predictions, [1.0] * learner_count, device
        )
        aggregations["J=1"] = reference.alpha_weighted_ensemble(
            [outer_predictions[0]], [1.0], device
        )
    rows = []
    for name, (prediction, _) in aggregations.items():
        metrics = evaluate_policy_ensemble(
            prediction,
            validation_bundle.y_outer_val_reg,
            validation_bundle.y_outer_val_cls,
            fee=float(config["fee"]),
            trading_freq=float(config["trading_freq"]),
        )
        rows.append(metric_row(dataset, variant, name, experiment_seed, "outer_validation", metrics, learner_count))

    if allow_final_test:
        test_aggregations: dict[str, tuple[dict[str, torch.Tensor], bool]] = {
            "Alpha-policy": reference.alpha_weighted_ensemble(final_predictions, alphas, device),
        }
        if learner_count > 1:
            test_aggregations["Equal-policy"] = reference.alpha_weighted_ensemble(
                final_predictions, [1.0] * learner_count, device
            )
            test_aggregations["J=1"] = reference.alpha_weighted_ensemble(
                [final_predictions[0]], [1.0], device
            )
        for name, (prediction, _) in test_aggregations.items():
            metrics = evaluate_policy_ensemble(
                prediction,
                validation_bundle.y_final_test_reg,
                validation_bundle.y_final_test_cls,
                fee=float(config["fee"]),
                trading_freq=float(config["trading_freq"]),
            )
            rows.append(metric_row(dataset, variant, name, experiment_seed, "final_test", metrics, learner_count))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full_hhi", "nosparse_hhi", "fixedk5"), default="full_hhi")
    parser.add_argument("--datasets", nargs="+", default=["DJIA30_2026", "SP500_TOP50_EX_PLTR_2026"])
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=3)
    parser.add_argument("--learner_count", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow_final_test", action="store_true")
    args = parser.parse_args()
    invalid = [name for name in args.datasets if name not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    if args.learner_count < 1 or args.learner_count > 5:
        raise ValueError("learner_count must be in [1,5]")
    if args.smoke and (args.learner_count != 1 or args.allow_final_test):
        raise ValueError("Smoke must be one learner with final test sealed")
    device = reference.resolve_device(args.device)
    rows = []
    for dataset in args.datasets:
        for experiment_seed in range(args.seed_start, args.seed_end + 1):
            rows.extend(
                run_seed(
                    variant=args.variant,
                    dataset=dataset,
                    experiment_seed=experiment_seed,
                    learner_count=args.learner_count,
                    device=device,
                    smoke=args.smoke,
                    allow_final_test=args.allow_final_test,
                )
            )
            print(f"[complete] {args.variant} {dataset} seed{experiment_seed:03d}", flush=True)
    scope = "formal" if args.allow_final_test else ("smoke" if args.smoke else "triage")
    output_dir = REPORT_ROOT / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_J{args.learner_count}_seed{args.seed_start:03d}_{args.seed_end:03d}"
    write_csv(output_dir / f"{stem}_per_seed.csv", rows)
    write_csv(output_dir / f"{stem}_summary.csv", summarize(rows))
    manifest = {
        "method": METHOD,
        "scope": scope,
        "variant": args.variant,
        "datasets": args.datasets,
        "seed_range": [args.seed_start, args.seed_end],
        "learner_count": args.learner_count,
        "deterministic": True,
        "final_test_access": "explicitly_unsealed" if args.allow_final_test else "sealed_not_loaded",
        "target_definition": execution_data.EXPECTED_TARGET,
        "config": config_for(args.variant),
        "output_root": str(OUTPUT_ROOT),
    }
    (output_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
