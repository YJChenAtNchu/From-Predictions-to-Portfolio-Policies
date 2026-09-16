from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


# CUBLAS determinism must be configured before the first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_data_execution_t1_t6_v2 as strict_data  # noqa: E402
from scripts import run_execution_t1_t6_proposed_20260902 as base  # noqa: E402
from torch_multiweak_ensemble import evaluate_policy_ensemble  # noqa: E402
from torch_portfolio import topk_softmax_weights  # noqa: E402


METHOD = "kbs_strict_v2_trial020_proposed"
PROTOCOL_ID = strict_data.EXPECTED_PROTOCOL_ID
SPLIT_PROTOCOL_ID = strict_data.SPLIT_PROTOCOL_ID
ACCOUNTING_PROTOCOL_ID = "drift_full_l1_entry_fee_mark_to_market_v1_20260905"
LABEL_VERSION = "adjusted_close_tplus6_over_tplus1_v2_20260905"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "kbs_revision_20260905" / "proposed"
REPORT_ROOT = PROJECT_ROOT / "outputs" / "kbs_revision_20260905" / "reports" / "proposed"
DATA_ROOT = PROJECT_ROOT / "data" / "execution_t1_t6_v2_20260905"
DATASETS = {
    "DJIA30_2026": DATA_ROOT / "DJIA30_2026",
    "SP500_TOP50_EX_PLTR_2026": DATA_ROOT / "SP500_TOP50_EX_PLTR_2026",
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED": (
        DATA_ROOT / "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_strict_seed(seed: int, deterministic: bool = True) -> None:
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
        torch.use_deterministic_algorithms(True)


def annotate_summary(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.get("meta", {})
    observed_protocol = meta.get("protocol_id", meta.get("dataset_summary", {}).get("protocol_id"))
    if observed_protocol != PROTOCOL_ID:
        raise ValueError(f"Strict protocol missing from {summary_path}")
    if meta.get("split_protocol_id") != SPLIT_PROTOCOL_ID:
        raise ValueError(f"Availability-purged split missing from {summary_path}")
    meta["protocol_id"] = PROTOCOL_ID
    summary["data_protocol"] = {
        "protocol_id": PROTOCOL_ID,
        "label_version": LABEL_VERSION,
        "outer_train": "execution-to-exit interval ends no later than 2020-12-31",
        "inner_train_validation": SPLIT_PROTOCOL_ID,
        "outer_validation": "2021-2023; reporting only in strict-v2 migration",
        "final_test": "2024-2025; sealed until checkpoint and alpha are frozen",
        "information_set": "adjusted OHLC through close[t]",
        "decision_time": "after close[t]",
        "execution": "adjusted close[t+1]",
        "target": strict_data.EXPECTED_TARGET,
    }
    audit = summary.setdefault("execution_t1_t6_audit", {})
    audit.update(
        {
            "method": METHOD,
            "protocol_id": PROTOCOL_ID,
            "split_protocol_id": SPLIT_PROTOCOL_ID,
            "accounting_protocol_id": ACCOUNTING_PROTOCOL_ID,
            "label_version": LABEL_VERSION,
            "target_definition": strict_data.EXPECTED_TARGET,
            "strict_v2_purge_present": True,
            "configuration_selection": (
                "historical trial020 architecture/configuration and HHI frozen before rerun"
            ),
            "outer_validation_used_for_checkpoint": False,
            "final_test_used_for_checkpoint": False,
            "deterministic_algorithms": "torch.use_deterministic_algorithms(True)",
            "deterministic_warn_only": False,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
    )
    summary["deterministic_seed_controls"] = [
        "PYTHONHASHSEED",
        "random.seed",
        "numpy.random.seed",
        "torch.manual_seed",
        "torch.cuda.manual_seed_all",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "cudnn.benchmark=False",
        "cudnn.deterministic=True",
        "torch.use_deterministic_algorithms(True)",
        "flash/memory-efficient SDP disabled",
    ]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def install_strict_bindings() -> None:
    base.METHOD = METHOD
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.REPORT_ROOT = REPORT_ROOT
    base.DATA_ROOT = DATA_ROOT
    base.DATASETS = DATASETS
    base.execution_data = strict_data
    base.reference.set_global_seed = set_strict_seed
    base.core.set_seed = set_strict_seed

    original_train = base.train_learner

    def train_and_annotate(**kwargs: Any) -> dict[str, Any]:
        report = original_train(**kwargs)
        directory = base.learner_dir(
            kwargs["variant"],
            kwargs["dataset"],
            kwargs["experiment_seed"],
            kwargs["learner_index"],
            kwargs["smoke"],
        )
        return annotate_summary(directory / "summary.json")

    base.train_learner = train_and_annotate

    original_final = base.evaluate_final_test

    def evaluate_and_annotate(**kwargs: Any) -> dict[str, np.ndarray]:
        outputs = original_final(**kwargs)
        directory = base.learner_dir(
            kwargs["variant"],
            kwargs["dataset"],
            kwargs["experiment_seed"],
            kwargs["learner_index"],
            False,
        )
        annotate_summary(directory / "summary.json")
        return outputs

    base.evaluate_final_test = evaluate_and_annotate


def _composition_dir(
    variant: str, dataset: str, experiment_seed: int, smoke: bool
) -> Path:
    first = base.learner_dir(variant, dataset, experiment_seed, 0, smoke)
    return first.parent / "compositions"


def export_compositions(
    *,
    variant: str,
    dataset: str,
    experiment_seed: int,
    learner_count: int,
    device: str,
    smoke: bool,
    split_prefix: str,
) -> None:
    predictions: list[dict[str, np.ndarray]] = []
    alphas: list[float] = []
    alpha_rounds: list[int] = []
    checkpoint_paths: list[str] = []
    for learner_index in range(learner_count):
        directory = base.learner_dir(
            variant, dataset, experiment_seed, learner_index, smoke
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        alpha, round_q = base.reference.best_checkpoint_alpha(summary)
        alphas.append(alpha)
        alpha_rounds.append(round_q)
        predictions.append(base.reference.load_prediction(directory, split_prefix))
        checkpoint_paths.append(summary["outputs"]["model"])

    config = base.config_for(variant)
    condition = base.condition_for(variant)
    allow_final = split_prefix == "final_test"
    bundle = base._load_bundle(
        base.DATASETS[dataset],
        condition,
        device,
        allow_final_test=allow_final,
        config=config,
    )
    if allow_final:
        y_reg = bundle.y_final_test_reg
        y_cls = bundle.y_final_test_cls
        split_name = "final_test"
    else:
        y_reg = bundle.y_outer_val_reg
        y_cls = bundle.y_outer_val_cls
        split_name = "outer_validation"

    alpha_prediction, alpha_fallback = base.reference.alpha_weighted_ensemble(
        predictions, alphas, device
    )
    equal_prediction, _ = base.reference.alpha_weighted_ensemble(
        predictions, [1.0] * learner_count, device
    )
    j1_prediction, _ = base.reference.alpha_weighted_ensemble(
        [predictions[0]], [1.0], device
    )
    score_prediction, score_fallback = base.reference.alpha_weighted_ensemble(
        predictions, alphas, device
    )
    score_weights, score_k, score_diag = topk_softmax_weights(
        score_prediction["reg_out"],
        k_max=int(config["k_max"]),
        k_min=int(config["k_min"]),
        decision_rule=str(config["decision_rule"]),
        eta_k=float(config["eta_k"]),
        eta_omega=float(config["eta_omega"]),
        return_diagnostics=True,
    )
    score_prediction["policy"] = score_weights

    fixed5_prediction = None
    fixed5_k = None
    fixed5_policy_bank = None
    if variant == "full_hhi":
        fixed5_predictions: list[dict[str, np.ndarray]] = []
        for prediction in predictions:
            learner_scores = torch.as_tensor(
                prediction["reg_out"], dtype=torch.float32, device=device
            )
            fixed_weights, learner_fixed_k, _ = topk_softmax_weights(
                learner_scores,
                k_max=5,
                k_min=5,
                decision_rule="fixed",
                eta_k=float(config["eta_k"]),
                eta_omega=float(config["eta_omega"]),
                return_diagnostics=True,
            )
            fixed_prediction = dict(prediction)
            fixed_prediction["policy"] = fixed_weights.detach().cpu().numpy()
            fixed5_predictions.append(fixed_prediction)
            fixed5_k = learner_fixed_k
        fixed5_prediction, fixed5_fallback = base.reference.alpha_weighted_ensemble(
            fixed5_predictions, alphas, device
        )
        fixed5_policy_bank = np.stack(
            [prediction["policy"] for prediction in fixed5_predictions], axis=0
        )

    compositions = {
        "Alpha-policy": (alpha_prediction, alpha_fallback, None, None),
        "Equal-policy": (equal_prediction, False, None, None),
        "Profit-score": (score_prediction, score_fallback, score_k, score_diag),
        "J1-policy": (j1_prediction, False, None, None),
    }
    if fixed5_prediction is not None:
        compositions["Fixed5-alpha-policy"] = (
            fixed5_prediction,
            fixed5_fallback,
            fixed5_k,
            None,
        )
    root = _composition_dir(variant, dataset, experiment_seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    if fixed5_policy_bank is not None:
        np.save(root / f"{split_name}_fixed5_learner_policy_bank.npy", fixed5_policy_bank)
    manifest_rows: list[dict[str, Any]] = []
    for name, (prediction, fallback, selected_k, diagnostics) in compositions.items():
        metrics = evaluate_policy_ensemble(
            prediction,
            y_reg,
            y_cls,
            fee=float(config["fee"]),
            trading_freq=float(config["trading_freq"]),
        )
        slug = name.lower().replace("-", "_")
        np.save(root / f"{split_name}_{slug}_weights.npy", prediction["policy"].detach().cpu().numpy())
        np.save(root / f"{split_name}_{slug}_reg_out.npy", prediction["reg_out"].detach().cpu().numpy())
        np.save(root / f"{split_name}_{slug}_cls_out.npy", prediction["cls_prob"].detach().cpu().numpy())
        np.save(root / f"{split_name}_{slug}_wealth.npy", metrics["_wealth"].detach().cpu().numpy())
        if selected_k is not None:
            np.save(root / f"{split_name}_{slug}_selected_k.npy", selected_k.detach().cpu().numpy())
            for key in ("hhi", "n_eff"):
                value = diagnostics.get(key) if diagnostics is not None else None
                if isinstance(value, torch.Tensor):
                    np.save(root / f"{split_name}_{slug}_{key}.npy", value.detach().cpu().numpy())
        manifest_rows.append(
            {
                "composition": name,
                "split": split_name,
                "used_equal_fallback": bool(fallback),
                "weights_file": str(root / f"{split_name}_{slug}_weights.npy"),
                "wealth_file": str(root / f"{split_name}_{slug}_wealth.npy"),
                "final_wealth": float(metrics["final_wealth"]),
                "total_turnover": float(metrics["total_turnover"]),
            }
        )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "accounting_protocol_id": ACCOUNTING_PROTOCOL_ID,
        "dataset": dataset,
        "variant": variant,
        "experiment_seed": experiment_seed,
        "learner_count": learner_count,
        "learner_alphas": alphas,
        "alpha_rounds": alpha_rounds,
        "checkpoint_paths": checkpoint_paths,
        "alpha_source": "training outcome at each inner-validation-selected checkpoint",
        "composition_weights_frozen_before_final_test": True,
        "records": manifest_rows,
    }
    (root / f"{split_name}_composition_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    install_strict_bindings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("full_hhi", "nosparse_hhi", "fixedk5"), default="full_hhi"
    )
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
    device = base.reference.resolve_device(args.device)

    # Phase 1 completes all fitting and freezes every checkpoint/alpha before
    # any final-test tensor is loaded.
    validation_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for experiment_seed in range(args.seed_start, args.seed_end + 1):
            validation_rows.extend(
                base.run_seed(
                    variant=args.variant,
                    dataset=dataset,
                    experiment_seed=experiment_seed,
                    learner_count=args.learner_count,
                    device=device,
                    smoke=args.smoke,
                    allow_final_test=False,
                )
            )
            export_compositions(
                variant=args.variant,
                dataset=dataset,
                experiment_seed=experiment_seed,
                learner_count=args.learner_count,
                device=device,
                smoke=args.smoke,
                split_prefix="outer_val",
            )
            print(f"[fit complete] {args.variant} {dataset} seed{experiment_seed:03d}", flush=True)

    rows = validation_rows
    if args.allow_final_test:
        rows = []
        # Phase 2 begins only after every requested learner has completed.
        for dataset in args.datasets:
            for experiment_seed in range(args.seed_start, args.seed_end + 1):
                rows.extend(
                    base.run_seed(
                        variant=args.variant,
                        dataset=dataset,
                        experiment_seed=experiment_seed,
                        learner_count=args.learner_count,
                        device=device,
                        smoke=False,
                        allow_final_test=True,
                    )
                )
                export_compositions(
                    variant=args.variant,
                    dataset=dataset,
                    experiment_seed=experiment_seed,
                    learner_count=args.learner_count,
                    device=device,
                    smoke=False,
                    split_prefix="final_test",
                )
                print(f"[frozen test complete] {args.variant} {dataset} seed{experiment_seed:03d}", flush=True)

    scope = "formal" if args.allow_final_test else ("smoke" if args.smoke else "fit_only")
    output_dir = REPORT_ROOT / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_J{args.learner_count}_seed{args.seed_start:03d}_{args.seed_end:03d}"
    base.write_csv(output_dir / f"{stem}_per_seed.csv", rows)
    base.write_csv(output_dir / f"{stem}_summary.csv", base.summarize(rows))
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": METHOD,
                "scope": scope,
                "variant": args.variant,
                "datasets": args.datasets,
                "seed_range": [args.seed_start, args.seed_end],
                "learner_count": args.learner_count,
                "deterministic": True,
                "final_test_access": "unsealed_after_all_fitting" if args.allow_final_test else "sealed_not_loaded",
                "target_definition": strict_data.EXPECTED_TARGET,
                "config": base.config_for(args.variant),
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for manifest_path in REPORT_ROOT.rglob("*_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "method": METHOD,
                "protocol_id": PROTOCOL_ID,
                "split_protocol_id": SPLIT_PROTOCOL_ID,
                "accounting_protocol_id": ACCOUNTING_PROTOCOL_ID,
                "label_version": LABEL_VERSION,
                "configuration_selection": "frozen before strict-v2 rerun",
                "final_test_selection_role": "none",
                "output_root": str(OUTPUT_ROOT),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
