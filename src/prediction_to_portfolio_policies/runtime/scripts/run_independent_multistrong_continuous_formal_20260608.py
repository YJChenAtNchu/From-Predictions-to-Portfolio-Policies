from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_data_formal_split import load_proposed_djia_formal_data  # noqa: E402
from torch_model import build_model_from_data  # noqa: E402
from torch_multiweak_ensemble import evaluate_policy_ensemble  # noqa: E402
from torch_portfolio import topk_softmax_weights  # noqa: E402
from torch_train_profitboost_formal_val import train_profitboost_formal_val  # noqa: E402
from torch_train_smoke import _take_batch, set_seed  # noqa: E402


OUT_ROOT = ROOT / "outputs" / "torch_proposed" / "formal_independent_multistrong_continuous_20260608"
REPORT_DIR = ROOT / "outputs" / "weekly_report" / "independent_multistrong_continuous_2026-06-08"

DATASETS: dict[str, str] = {
    "DJIA30_2026": "data/djia30_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
    "SP500_TOP50_EX_PLTR_2026": "data/sp500_top50_ex_pltr_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
}


def learner_seed(experiment_seed: int, learner_index: int) -> int:
    return int(experiment_seed) * 1000 + int(learner_index) + 1


def ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]
    return array


def train_or_load_learner(
    *,
    dataset_name: str,
    data_root: str,
    experiment_seed: int,
    learner_index: int,
    max_outer_rounds: int,
    inner_epochs: int,
    inner_patience: int,
    outer_patience: int,
    device: str,
) -> dict[str, Any]:
    current_seed = learner_seed(experiment_seed, learner_index)
    learner_dir = (
        OUT_ROOT
        / dataset_name
        / f"experiment_seed{experiment_seed:03d}"
        / f"learner{learner_index + 1:02d}_seed{current_seed:05d}"
    )
    summary_path = learner_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(
        f"[independent multistrong] dataset={dataset_name} "
        f"experiment_seed={experiment_seed:03d} learner={learner_index + 1} "
        f"learner_seed={current_seed} start",
        flush=True,
    )
    report = train_profitboost_formal_val(
        data_root=data_root,
        out_dir=learner_dir,
        seed=current_seed,
        max_outer_rounds=max_outer_rounds,
        inner_epochs=inner_epochs,
        inner_patience=inner_patience,
        outer_patience=outer_patience,
        batch_size=16,
        lr=1e-5,
        weight_decay=0.0,
        beta=0.6,
        ce_weight=1.0,
        mse_weight=1.0,
        loss_mode="ce_mse_rank",
        rank_weight=1.0,
        use_contrastive=False,
        use_sharpe=False,
        use_ema_loss_norm=True,
        ema_alpha=0.1,
        fee=0.001,
        k_max=5,
        k_min=1,
        decision_rule="spread",
        eta_k=1.0,
        eta_omega=1.0,
        trading_freq=5.0,
        frontend_type="lista",
        lista_unroll_steps=4,
        lbp_unroll_steps=4,
        dropout_rate=0.1,
        weight_update_mode="sample_specific",
        profit_source="one_step_sum",
        inner_train_val_split=0.9,
        log_every_epoch=0,
        device=device,
    )
    torch.cuda.empty_cache()
    return report


def load_outer_val_prediction(learner_dir: Path) -> dict[str, np.ndarray]:
    return {
        "policy": ensure_2d(np.load(learner_dir / "outer_val_weights.npy")).astype(np.float32),
        "cls_prob": np.load(learner_dir / "outer_val_cls_out.npy").astype(np.float32),
        "reg_out": np.load(learner_dir / "outer_val_reg_out.npy").astype(np.float32),
    }


@torch.no_grad()
def evaluate_checkpoint_final_test(
    *,
    learner_dir: Path,
    data_root: str,
    device: str,
) -> dict[str, np.ndarray]:
    paths = {
        "policy": learner_dir / "final_test_weights.npy",
        "cls_prob": learner_dir / "final_test_cls_out.npy",
        "reg_out": learner_dir / "final_test_reg_out.npy",
    }
    if all(path.exists() for path in paths.values()):
        return {
            key: ensure_2d(np.load(path)).astype(np.float32) if key == "policy" else np.load(path).astype(np.float32)
            for key, path in paths.items()
        }

    summary = json.loads((learner_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    seed = int(config["seed"])
    set_seed(seed)
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_proposed_djia_formal_data(
        root=data_root,
        inner_train_val_split=float(config.get("inner_train_val_split", 0.9)),
        device=resolved_device,
        use_signature=bool(config.get("use_signature", False)),
        signature_level=int(config.get("signature_level") or 2),
    )
    model = build_model_from_data(
        bundle,
        frontend_type=str(config.get("frontend_type", "lista")),
        lista_unroll_steps=int(config.get("lista_unroll_steps", 4)),
        lbp_unroll_steps=int(config.get("lbp_unroll_steps", 4)),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
    ).to(resolved_device)
    warmup_idx = torch.arange(min(int(config.get("batch_size", 16)), bundle.y_train_reg.shape[0]), device=resolved_device)
    model(_take_batch(bundle.x_train_list, warmup_idx))
    model_path = learner_dir / f"model_seed{seed:03d}.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location=resolved_device))
    model.eval()
    outputs = model(bundle.x_final_test_list)
    policy, _ = topk_softmax_weights(
        outputs["reg_out"],
        k_min=int(config.get("k_min", 1)),
        k_max=int(config.get("k_max", 5)),
        decision_rule=str(config.get("decision_rule", "spread")),
        eta_k=float(config.get("eta_k", 1.0)),
        eta_omega=float(config.get("eta_omega", 1.0)),
    )
    result = {
        "policy": policy.detach().cpu().numpy().astype(np.float32),
        "cls_prob": outputs["cls_out"].detach().cpu().numpy().astype(np.float32),
        "reg_out": outputs["reg_out"].detach().cpu().numpy().astype(np.float32),
    }
    np.save(paths["policy"], result["policy"])
    np.save(paths["cls_prob"], result["cls_prob"])
    np.save(paths["reg_out"], result["reg_out"])
    torch.cuda.empty_cache()
    return result


def materialize_equal_ensemble(predictions: list[dict[str, np.ndarray]], device: str) -> dict[str, torch.Tensor]:
    policy = torch.as_tensor(np.mean([p["policy"] for p in predictions], axis=0), dtype=torch.float32, device=device)
    cls_prob = torch.as_tensor(np.mean([p["cls_prob"] for p in predictions], axis=0), dtype=torch.float32, device=device)
    reg_out = torch.as_tensor(np.mean([p["reg_out"] for p in predictions], axis=0), dtype=torch.float32, device=device)
    return {
        "policy": policy,
        "cls_prob": cls_prob,
        "reg_out": reg_out,
        "alpha_total": torch.as_tensor(float(len(predictions)), dtype=torch.float32, device=device),
        "used_fallback": torch.zeros((), dtype=torch.bool, device=device),
    }


def metric_row(
    *,
    dataset_name: str,
    experiment_seed: int,
    selected_j: int,
    metrics: dict[str, Any],
    stage: str,
    note: str,
) -> dict[str, Any]:
    fin = metrics["financial_metrics"]
    return {
        "dataset": dataset_name,
        "experiment_seed": experiment_seed,
        "stage": stage,
        "selected_j": selected_j,
        "ACC": metrics["acc"],
        "APV": fin["APV"],
        "ARR": fin["APR"],
        "AVol": fin["AVol"],
        "ASR": fin["ASR"],
        "MDD": fin["MDD"],
        "CR": fin["CR"],
        "Turnover": metrics["total_turnover"],
        "profit_on_1usd": metrics["profit_on_1usd"],
        "mean_support_size": metrics.get("mean_support_size"),
        "note": note,
    }


def run_experiment_seed(
    *,
    dataset_name: str,
    data_root: str,
    experiment_seed: int,
    max_learners: int,
    ensemble_patience: int,
    max_outer_rounds: int,
    inner_epochs: int,
    inner_patience: int,
    outer_patience: int,
    device: str,
) -> dict[str, Any]:
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_proposed_djia_formal_data(root=data_root, inner_train_val_split=0.9, device=resolved_device)
    experiment_dir = OUT_ROOT / dataset_name / f"experiment_seed{experiment_seed:03d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_dir / "independent_multistrong_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    outer_predictions: list[dict[str, np.ndarray]] = []
    learner_records: list[dict[str, Any]] = []
    best_outer_profit = -float("inf")
    best_j = 0
    bad_rounds = 0

    for learner_index in range(max_learners):
        report = train_or_load_learner(
            dataset_name=dataset_name,
            data_root=data_root,
            experiment_seed=experiment_seed,
            learner_index=learner_index,
            max_outer_rounds=max_outer_rounds,
            inner_epochs=inner_epochs,
            inner_patience=inner_patience,
            outer_patience=outer_patience,
            device=device,
        )
        learner_seed_value = int(report["config"]["seed"])
        learner_dir = (
            OUT_ROOT
            / dataset_name
            / f"experiment_seed{experiment_seed:03d}"
            / f"learner{learner_index + 1:02d}_seed{learner_seed_value:05d}"
        )
        outer_predictions.append(load_outer_val_prediction(learner_dir))
        ensemble = materialize_equal_ensemble(outer_predictions, resolved_device)
        outer_metrics = evaluate_policy_ensemble(
            ensemble,
            bundle.y_outer_val_reg,
            bundle.y_outer_val_cls,
            fee=0.001,
            trading_freq=5.0,
        )
        outer_profit = float(outer_metrics["profit_on_1usd"])
        improved = outer_profit > best_outer_profit
        if improved:
            best_outer_profit = outer_profit
            best_j = learner_index + 1
            bad_rounds = 0
        else:
            bad_rounds += 1

        learner_records.append(
            {
                "learner_index": learner_index + 1,
                "learner_seed": learner_seed_value,
                "outer_val_profit_on_1usd": outer_profit,
                "outer_val_APV": outer_metrics["financial_metrics"]["APV"],
                "outer_val_ARR": outer_metrics["financial_metrics"]["APR"],
                "outer_val_MDD": outer_metrics["financial_metrics"]["MDD"],
                "outer_val_CR": outer_metrics["financial_metrics"]["CR"],
                "selected_as_best_j": improved,
                "bad_rounds": bad_rounds,
            }
        )
        print(
            f"[independent multistrong] dataset={dataset_name} seed={experiment_seed:03d} "
            f"J={learner_index + 1} outer_profit={outer_profit:.6f} "
            f"best_j={best_j} bad_rounds={bad_rounds}/{ensemble_patience}",
            flush=True,
        )
        if bad_rounds >= ensemble_patience:
            break

    selected_predictions = outer_predictions[:best_j]
    selected_outer_ensemble = materialize_equal_ensemble(selected_predictions, resolved_device)
    selected_outer_metrics = evaluate_policy_ensemble(
        selected_outer_ensemble,
        bundle.y_outer_val_reg,
        bundle.y_outer_val_cls,
        fee=0.001,
        trading_freq=5.0,
    )

    final_predictions = []
    selected_learner_seeds = []
    for learner_index in range(best_j):
        learner_seed_value = learner_seed(experiment_seed, learner_index)
        learner_dir = (
            OUT_ROOT
            / dataset_name
            / f"experiment_seed{experiment_seed:03d}"
            / f"learner{learner_index + 1:02d}_seed{learner_seed_value:05d}"
        )
        final_predictions.append(
            evaluate_checkpoint_final_test(
                learner_dir=learner_dir,
                data_root=data_root,
                device=device,
            )
        )
        selected_learner_seeds.append(learner_seed_value)

    selected_final_ensemble = materialize_equal_ensemble(final_predictions, resolved_device)
    selected_final_metrics = evaluate_policy_ensemble(
        selected_final_ensemble,
        bundle.y_final_test_reg,
        bundle.y_final_test_cls,
        fee=0.001,
        trading_freq=5.0,
    )
    np.save(
        experiment_dir / "final_test_ensemble_policy.npy",
        selected_final_ensemble["policy"].detach().cpu().numpy(),
    )
    np.save(
        experiment_dir / "final_test_ensemble_cls_prob.npy",
        selected_final_ensemble["cls_prob"].detach().cpu().numpy(),
    )
    np.save(
        experiment_dir / "final_test_ensemble_reg_out.npy",
        selected_final_ensemble["reg_out"].detach().cpu().numpy(),
    )

    summary = {
        "method": "Independent Multi-Strong Continuous Ensemble",
        "dataset": dataset_name,
        "experiment_seed": experiment_seed,
        "max_learners": max_learners,
        "ensemble_patience": ensemble_patience,
        "selected_j": best_j,
        "selected_learner_seeds": selected_learner_seeds,
        "config": {
            "base_learner": "Proposed B continuous",
            "each_j_starts_from": "fresh model + W0",
            "cross_learner_weight_sharing": False,
            "aggregation": "equal average in policy space",
            "inner_patience": inner_patience,
            "outer_patience": outer_patience,
            "max_outer_rounds_per_learner": max_outer_rounds,
            "inner_epochs": inner_epochs,
            "ensemble_selection_metric": "outer_validation profit_on_1usd",
        },
        "learner_records": learner_records,
        "outer_validation": metric_row(
            dataset_name=dataset_name,
            experiment_seed=experiment_seed,
            selected_j=best_j,
            metrics=selected_outer_metrics,
            stage="outer_validation",
            note="selected J by outer validation",
        ),
        "final_test": metric_row(
            dataset_name=dataset_name,
            experiment_seed=experiment_seed,
            selected_j=best_j,
            metrics=selected_final_metrics,
            stage="final_test",
            note="fixed selected J from outer validation",
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["stage"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, stage), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "dataset": dataset,
            "stage": stage,
            "n_seeds": len(group),
            "positive_seeds": sum(1 for row in group if float(row["APV"]) > 1.0),
        }
        for metric in ["selected_j", "ACC", "APV", "ARR", "AVol", "ASR", "MDD", "CR", "Turnover"]:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out.append(summary)
    return out


def build_report(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    def fmt(value: float, pct: bool = False) -> str:
        return f"{value * 100:.2f}%" if pct else f"{value:.3f}"

    trs = []
    for row in summary_rows:
        trs.append(
            "<tr>"
            f"<td>{row['dataset']}</td><td>{row['stage']}</td><td>{row['n_seeds']}</td><td>{row['positive_seeds']}</td>"
            f"<td>{row['selected_j_mean']:.2f} ± {row['selected_j_std']:.2f}</td>"
            f"<td>{fmt(row['ACC_mean'])} ± {fmt(row['ACC_std'])}</td>"
            f"<td>{fmt(row['APV_mean'])} ± {fmt(row['APV_std'])}</td>"
            f"<td>{fmt(row['ARR_mean'], True)} ± {fmt(row['ARR_std'], True)}</td>"
            f"<td>{fmt(row['ASR_mean'])} ± {fmt(row['ASR_std'])}</td>"
            f"<td>{fmt(row['MDD_mean'], True)} ± {fmt(row['MDD_std'], True)}</td>"
            f"<td>{fmt(row['CR_mean'])} ± {fmt(row['CR_std'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Independent Multi-Strong Continuous Ensemble</title>
  <style>
    body {{ font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; margin: 28px; line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d5dde5; padding: 7px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #edf4f8; }}
  </style>
</head>
<body>
  <h1>Independent Multi-Strong Continuous Ensemble</h1>
  <p>
    每個 learner j 都是 fresh model + W0，並在 j 內部完整執行 Proposed B continuous ProfitBoost。
    learner 之間不共享 sample weight；J 由 outer validation ensemble profit early stopping 自動決定。
  </p>
  <table>
    <thead>
      <tr><th>Dataset</th><th>Stage</th><th>N</th><th>Positive</th><th>Selected J</th><th>ACC</th><th>APV</th><th>ARR</th><th>ASR</th><th>MDD</th><th>CR</th></tr>
    </thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_learners", type=int, default=10)
    parser.add_argument("--ensemble_patience", type=int, default=5)
    parser.add_argument("--max_outer_rounds", type=int, default=50)
    parser.add_argument("--inner_epochs", type=int, default=1000)
    parser.add_argument("--inner_patience", type=int, default=5)
    parser.add_argument("--outer_patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset_name, data_root in DATASETS.items():
        for experiment_seed in range(1, args.seeds + 1):
            summary = run_experiment_seed(
                dataset_name=dataset_name,
                data_root=data_root,
                experiment_seed=experiment_seed,
                max_learners=args.max_learners,
                ensemble_patience=args.ensemble_patience,
                max_outer_rounds=args.max_outer_rounds,
                inner_epochs=args.inner_epochs,
                inner_patience=args.inner_patience,
                outer_patience=args.outer_patience,
                device=args.device,
            )
            rows.append(summary["outer_validation"])
            rows.append(summary["final_test"])
            summary_rows = summarize(rows)
            write_csv(REPORT_DIR / "independent_multistrong_continuous_per_seed.csv", rows)
            write_csv(REPORT_DIR / "independent_multistrong_continuous_group_summary.csv", summary_rows)
            (REPORT_DIR / "independent_multistrong_continuous_report.html").write_text(
                build_report(rows, summary_rows), encoding="utf-8"
            )
            (REPORT_DIR / "independent_multistrong_continuous_summary.json").write_text(
                json.dumps({"rows": rows, "summary": summary_rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(json.dumps({"output": str(REPORT_DIR), "n_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
