from __future__ import annotations

import argparse
import csv
import html
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
from torch_multiweak_ensemble import evaluate_policy_ensemble  # noqa: E402
from scripts.run_independent_multistrong_continuous_formal_20260608 import (  # noqa: E402
    evaluate_checkpoint_final_test,
)


SOURCE_ROOT = ROOT / "outputs" / "torch_proposed" / "formal_independent_multistrong_continuous_20260608"
OUT_DIR = ROOT / "outputs" / "weekly_report" / "independent_multistrong_alpha_fixedj5_2026-06-08"
MAIN_REPORT = ROOT / "outputs" / "weekly_report" / "formal_djia30_sp500_top50_comparison_2026-06-04.html"
SECTION_START = "<!-- INDEPENDENT_MULTISTRONG_ALPHA_FIXEDJ5_START -->"
SECTION_END = "<!-- INDEPENDENT_MULTISTRONG_ALPHA_FIXEDJ5_END -->"
FIXED_J = 5

DATASETS = {
    "DJIA30_2026": ROOT / "data" / "djia30_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
    "SP500_TOP50_EX_PLTR_2026": ROOT
    / "data"
    / "sp500_top50_ex_pltr_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
}


def ensure_policy(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]
    return array


def best_checkpoint_alpha(summary: dict[str, Any]) -> tuple[float, int]:
    selected = [row for row in summary["outer_log"] if bool(row.get("selected_as_best"))]
    if not selected:
        raise ValueError("Learner summary has no selected_as_best outer round")
    best_row = selected[-1]
    return max(float(best_row["alpha_weight"]), 0.0), int(best_row["outer_round"])


def load_prediction(learner_dir: Path, split_prefix: str) -> dict[str, np.ndarray]:
    return {
        "policy": ensure_policy(np.load(learner_dir / f"{split_prefix}_weights.npy")).astype(np.float32),
        "cls_prob": np.load(learner_dir / f"{split_prefix}_cls_out.npy").astype(np.float32),
        "reg_out": np.load(learner_dir / f"{split_prefix}_reg_out.npy").astype(np.float32),
    }


def alpha_weighted_ensemble(
    predictions: list[dict[str, np.ndarray]],
    alphas: list[float],
    device: str,
) -> tuple[dict[str, torch.Tensor], bool]:
    alpha_array = np.asarray(alphas, dtype=np.float64)
    used_equal_fallback = float(alpha_array.sum()) <= 1e-12
    if used_equal_fallback:
        normalized = np.ones(len(alphas), dtype=np.float64) / len(alphas)
    else:
        normalized = alpha_array / alpha_array.sum()

    def weighted(key: str) -> torch.Tensor:
        value = np.sum(
            np.stack([prediction[key] for prediction in predictions], axis=0)
            * normalized.reshape((-1,) + (1,) * predictions[0][key].ndim),
            axis=0,
        )
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    return {
        "policy": weighted("policy"),
        "cls_prob": weighted("cls_prob"),
        "reg_out": weighted("reg_out"),
        "alpha_total": torch.as_tensor(float(alpha_array.sum()), dtype=torch.float32, device=device),
        "used_fallback": torch.as_tensor(used_equal_fallback, dtype=torch.bool, device=device),
    }, used_equal_fallback


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def metric_row(
    dataset: str,
    experiment_seed: int,
    split: str,
    metrics: dict[str, Any],
    alphas: list[float],
    alpha_rounds: list[int],
    fallback: bool,
) -> dict[str, Any]:
    financial = metrics["financial_metrics"]
    alpha_array = np.asarray(alphas, dtype=np.float64)
    normalized = (
        alpha_array / alpha_array.sum()
        if alpha_array.sum() > 1e-12
        else np.ones(FIXED_J, dtype=np.float64) / FIXED_J
    )
    return {
        "dataset": dataset,
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
        "equal_weight_fallback": fallback,
        "alphas": ",".join(f"{value:.10f}" for value in alphas),
        "alpha_outer_rounds": ",".join(str(value) for value in alpha_rounds),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for split in ["outer_validation", "final_test"]:
            group = [row for row in rows if row["dataset"] == dataset and row["split"] == split]
            summary: dict[str, Any] = {
                "dataset": dataset,
                "split": split,
                "n_seeds": len(group),
                "positive_seeds": sum(float(row["APV"]) > 1.0 for row in group),
                "fixed_J": FIXED_J,
                "fallback_seeds": sum(_as_bool(row["equal_weight_fallback"]) for row in group),
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
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            summaries.append(summary)
    return summaries


def fmt(mean: float, std: float, percent: bool = False) -> str:
    if percent:
        return f"{mean * 100:.2f}% ± {std * 100:.2f}%"
    return f"{mean:.3f} ± {std:.3f}"


def build_section(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for row in summaries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['dataset'])}</td>"
            f"<td>{html.escape(row['split'])}</td>"
            f"<td>{row['n_seeds']}</td>"
            f"<td>{row['positive_seeds']}/{row['n_seeds']}</td>"
            f"<td>{fmt(row['ACC_mean'], row['ACC_std'])}</td>"
            f"<td>{fmt(row['APV_mean'], row['APV_std'])}</td>"
            f"<td>{fmt(row['ARR_mean'], row['ARR_std'], True)}</td>"
            f"<td>{fmt(row['ASR_mean'], row['ASR_std'])}</td>"
            f"<td>{fmt(row['MDD_mean'], row['MDD_std'], True)}</td>"
            f"<td>{fmt(row['CR_mean'], row['CR_std'])}</td>"
            f"<td>{fmt(row['effective_learner_number_mean'], row['effective_learner_number_std'])}</td>"
            f"<td>{fmt(row['normalized_alpha_max_mean'], row['normalized_alpha_max_std'])}</td>"
            "</tr>"
        )
    return f"""
<section>
  <h2>Independent Multi-Strong Continuous：Alpha-weighted Fixed J=5</h2>
  <p>
    每個 learner 僅使用 Outer Train 內的資料完成 continuous ProfitBoost。
    外層 alpha 取自該 learner 最佳 checkpoint 所對應的 Inner Training profit alpha。
    J 固定為 5；Outer Validation 與 Final Test 均只作評估，不用來重算 alpha、更新 W 或選 J。
  </p>
  <p><code>final_policy = sum(alpha_j * policy_j) / sum(alpha_j)</code></p>
  <table>
    <thead><tr>
      <th>Dataset</th><th>Split</th><th>Seeds</th><th>Positive</th><th>ACC</th>
      <th>APV</th><th>ARR</th><th>ASR</th><th>MDD</th><th>CR</th>
      <th>Effective Learners</th><th>Max Normalized Alpha</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def update_main_report(section: str) -> None:
    text = MAIN_REPORT.read_text(encoding="utf-8", errors="replace")
    wrapped = f"{SECTION_START}\n{section}\n{SECTION_END}"
    if SECTION_START in text and SECTION_END in text:
        start = text.index(SECTION_START)
        end = text.index(SECTION_END) + len(SECTION_END)
        text = text[:start] + wrapped + text[end:]
    else:
        index = text.lower().rfind("</body>")
        text = text[:index] + wrapped + "\n" + text[index:]
    MAIN_REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate alpha-weighted Independent Multi-Strong Fixed J ensemble."
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Evaluate all datasets or one completed dataset.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(OUT_DIR),
        help="Output directory. Use a separate directory for partial summaries.",
    )
    parser.add_argument("--skip_main_report", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}

    selected_datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for dataset, data_root in selected_datasets.items():
        bundle = load_proposed_djia_formal_data(root=data_root, inner_train_val_split=0.9, device=device)
        detail[dataset] = {}
        for experiment_seed in range(1, args.seeds + 1):
            experiment_dir = SOURCE_ROOT / dataset / f"experiment_seed{experiment_seed:03d}"
            learner_dirs = sorted(experiment_dir.glob("learner*_seed*"))[:FIXED_J]
            if len(learner_dirs) != FIXED_J:
                raise FileNotFoundError(
                    f"{dataset} seed{experiment_seed:03d} has {len(learner_dirs)} learners; expected {FIXED_J}"
                )

            alphas: list[float] = []
            alpha_rounds: list[int] = []
            outer_predictions: list[dict[str, np.ndarray]] = []
            final_predictions: list[dict[str, np.ndarray]] = []
            for learner_dir in learner_dirs:
                summary = json.loads((learner_dir / "summary.json").read_text(encoding="utf-8"))
                alpha, alpha_round = best_checkpoint_alpha(summary)
                alphas.append(alpha)
                alpha_rounds.append(alpha_round)
                outer_predictions.append(load_prediction(learner_dir, "outer_val"))
                final_paths = [
                    learner_dir / "final_test_weights.npy",
                    learner_dir / "final_test_cls_out.npy",
                    learner_dir / "final_test_reg_out.npy",
                ]
                if all(path.exists() for path in final_paths):
                    final_predictions.append(load_prediction(learner_dir, "final_test"))
                else:
                    final_predictions.append(
                        evaluate_checkpoint_final_test(
                            learner_dir=learner_dir,
                            data_root=str(data_root),
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
            rows.append(
                metric_row(
                    dataset,
                    experiment_seed,
                    "outer_validation",
                    outer_metrics,
                    alphas,
                    alpha_rounds,
                    outer_fallback,
                )
            )
            rows.append(
                metric_row(
                    dataset,
                    experiment_seed,
                    "final_test",
                    final_metrics,
                    alphas,
                    alpha_rounds,
                    final_fallback,
                )
            )
            detail[dataset][f"seed{experiment_seed:03d}"] = {
                "learner_dirs": [str(path) for path in learner_dirs],
                "alphas": alphas,
                "alpha_outer_rounds": alpha_rounds,
                "outer_validation": clean_metrics(outer_metrics),
                "final_test": clean_metrics(final_metrics),
            }

    summaries = summarize(rows)
    write_csv(out_dir / "alpha_fixedj5_per_seed.csv", rows)
    write_csv(out_dir / "alpha_fixedj5_group_summary.csv", summaries)
    payload = {
        "method": "Independent Multi-Strong Continuous Alpha-Weighted Fixed J=5",
        "protocol": {
            "J": FIXED_J,
            "n_experiment_seeds": int(args.seeds),
            "alpha_source": "best checkpoint outer round alpha computed from Inner Training profit",
            "outer_validation_role": "evaluation only",
            "final_test_role": "evaluation only",
            "cross_learner_sample_weight_sharing": False,
        },
        "summary": summaries,
        "detail": detail,
    }
    (out_dir / "alpha_fixedj5_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    section = build_section(summaries)
    (out_dir / "alpha_fixedj5_report.html").write_text(
        f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        f'<title>Alpha-weighted Fixed J=5</title></head><body>{section}</body></html>',
        encoding="utf-8",
    )
    if not args.skip_main_report:
        update_main_report(section)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
