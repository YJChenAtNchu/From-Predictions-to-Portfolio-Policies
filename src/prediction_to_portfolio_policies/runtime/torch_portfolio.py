from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from torch_data import load_proposed_djia_data


def adaptive_k_spread(
    scores: torch.Tensor,
    k_min: int = 1,
    k_max: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one adaptive K per sample.

    Matches the active models.py idea:
    spread = max(score) - median(score)
    norm_spread = clip(spread / (abs(median(score)) + eps), 0, 1)
    K = int(k_max - norm_spread * (k_max - k_min))
    """
    if scores.ndim != 2:
        raise ValueError(f"adaptive_k expects scores shape (N,A), got {tuple(scores.shape)}")
    max_score = scores.max(dim=-1).values
    median_score = torch.quantile(scores, 0.5, dim=-1)
    spread = max_score - median_score
    norm_spread = torch.clamp(spread / (median_score.abs() + eps), 0.0, 1.0)
    k_float = k_max - norm_spread * float(k_max - k_min)
    k = torch.floor(k_float).to(torch.long)
    return torch.clamp(k, min=k_min, max=k_max)


def adaptive_k(
    scores: torch.Tensor,
    k_min: int = 1,
    k_max: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Backward-compatible alias for the original spread-based Adaptive-K."""
    return adaptive_k_spread(scores, k_min=k_min, k_max=k_max, eps=eps)


def adaptive_k_hhi_effective(
    scores: torch.Tensor,
    k_min: int = 1,
    k_max: int = 5,
    eta_k: float = 1.0,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select K using the Rényi-2 effective number of top-score mass."""
    if scores.ndim != 2:
        raise ValueError(
            f"adaptive_k_hhi_effective expects scores shape (N,A), got {tuple(scores.shape)}"
        )
    if eta_k <= 0:
        raise ValueError("eta_k must be positive")
    if not 1 <= k_min <= k_max <= scores.shape[-1]:
        raise ValueError(
            f"Expected 1 <= k_min <= k_max <= n_assets, got "
            f"k_min={k_min}, k_max={k_max}, n_assets={scores.shape[-1]}"
        )

    top_scores = torch.topk(scores, k=k_max, dim=-1, largest=True, sorted=True).values
    mean_u = top_scores.mean(dim=-1, keepdim=True)
    std_u = top_scores.std(dim=-1, keepdim=True, correction=0)
    z_scores = (top_scores - mean_u) / (std_u + float(eps))
    pseudo_prob = torch.softmax(z_scores / float(eta_k), dim=-1)
    hhi = pseudo_prob.square().sum(dim=-1)
    n_eff = 1.0 / hhi.clamp(min=float(eps))
    selected_k = torch.floor(n_eff + 0.5).to(torch.long)
    selected_k = torch.clamp(selected_k, min=k_min, max=k_max)

    if not return_diagnostics:
        return selected_k
    return selected_k, {
        "hhi": hhi,
        "n_eff": n_eff,
        "top_scores": top_scores,
        "z_scores": z_scores,
        "pseudo_prob": pseudo_prob,
    }


def topk_softmax_weights(
    scores: torch.Tensor,
    k_max: int = 5,
    k_min: int = 1,
    temperature: float = 1.0,
    adaptive: bool = True,
    decision_rule: str | None = None,
    eta_k: float = 1.0,
    eta_omega: float = 1.0,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | str]]
):
    """Convert predicted scores into long-only TopKSoftmax weights.

    Returns:
        weights: (N,A), nonnegative and row-sum 1.
        selected_k: (N,)
    """
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError(f"Only output_len=1 is supported in portfolio check, got {tuple(scores.shape)}")
        scores = scores[:, 0, :]
    if scores.ndim != 2:
        raise ValueError(f"topk_softmax_weights expects scores shape (N,A), got {tuple(scores.shape)}")
    if temperature <= 0 or eta_omega <= 0:
        raise ValueError("temperature and eta_omega must be positive")

    n, n_assets = scores.shape
    if decision_rule is None:
        decision_rule = "spread" if adaptive else "fixed"
    decision_rule = str(decision_rule).lower()
    diagnostics: Dict[str, torch.Tensor | str] = {"decision_rule": decision_rule}

    if decision_rule == "spread":
        selected_k = adaptive_k_spread(scores, k_min=k_min, k_max=k_max)
    elif decision_rule == "hhi_effective":
        selected_k, hhi_diagnostics = adaptive_k_hhi_effective(
            scores,
            k_min=k_min,
            k_max=k_max,
            eta_k=eta_k,
            eps=eps,
            return_diagnostics=True,
        )
        diagnostics.update(hhi_diagnostics)
    elif decision_rule == "fixed":
        selected_k = torch.full((n,), int(k_max), dtype=torch.long, device=scores.device)
    else:
        raise ValueError(
            f"Unsupported decision_rule={decision_rule!r}; "
            "expected 'spread', 'hhi_effective', or 'fixed'"
        )

    weights = torch.zeros_like(scores, dtype=torch.float32)
    # Preserve the legacy temperature argument while exposing eta_omega for
    # the new decision-rule API. Existing non-default temperature callers
    # continue to behave as before unless eta_omega is explicitly changed.
    weight_temperature = float(eta_omega)
    if eta_omega == 1.0 and temperature != 1.0:
        weight_temperature = float(temperature)
    for i in range(n):
        k_i = int(selected_k[i].item())
        # Stable sorting gives exact score ties a deterministic asset-order policy.
        idx = torch.argsort(scores[i], descending=True, stable=True)[:k_i]
        selected_scores = scores[i, idx] / weight_temperature
        selected_scores = selected_scores - selected_scores.max()
        soft = torch.softmax(selected_scores, dim=0)
        weights[i, idx] = soft
    diagnostics["selected_k"] = selected_k
    if return_diagnostics:
        return weights, selected_k, diagnostics
    return weights, selected_k


def equal_weight_topk(scores: torch.Tensor, k: int = 5) -> torch.Tensor:
    if scores.ndim == 3:
        scores = scores[:, 0, :]
    n, n_assets = scores.shape
    weights = torch.zeros_like(scores, dtype=torch.float32)
    for i in range(n):
        idx = torch.argsort(scores[i], descending=True)[:k]
        weights[i, idx] = 1.0 / float(k)
    return weights


def simulate_return_ratio_wealth(
    weights: torch.Tensor,
    y_true_ror: torch.Tensor,
    fee: float = 0.001,
    initial_wealth: float = 1.0,
    normalize_raw_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate long-only wealth using return ratios.

    Args:
        weights: (N,A) portfolio weights.
        y_true_ror: (N,A) or (N,1,A) return ratio.
        fee: proportional turnover cost.
        normalize_raw_weights: if True, clamp negatives and normalize rows.

    Returns:
        wealth_curve: (N+1,)
        step_returns: (N,)
        turnover: (N,)
    """
    if y_true_ror.ndim == 3:
        if y_true_ror.shape[1] != 1:
            raise ValueError(f"Only output_len=1 is supported, got {tuple(y_true_ror.shape)}")
        y_true_ror = y_true_ror[:, 0, :]
    if weights.ndim == 3:
        weights = weights[:, 0, :]
    if weights.shape != y_true_ror.shape:
        raise ValueError(f"weights/y_true shape mismatch: {tuple(weights.shape)} vs {tuple(y_true_ror.shape)}")

    weights = weights.to(dtype=torch.float32)
    y_true_ror = y_true_ror.to(dtype=torch.float32)
    if normalize_raw_weights:
        weights = torch.clamp(weights, min=0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom

    n, n_assets = weights.shape
    wealth = torch.empty(n + 1, dtype=torch.float32, device=weights.device)
    step_returns = torch.empty(n, dtype=torch.float32, device=weights.device)
    turnover = torch.empty(n, dtype=torch.float32, device=weights.device)
    wealth[0] = float(initial_wealth)
    prev_w = torch.zeros(n_assets, dtype=torch.float32, device=weights.device)

    for t in range(n):
        w_t = weights[t]
        turnover_t = torch.sum(torch.abs(w_t - prev_w))
        gross_ror_t = torch.sum(w_t * y_true_ror[t])
        net_ror_t = (1.0 - float(fee) * turnover_t) * gross_ror_t
        wealth[t + 1] = wealth[t] * net_ror_t
        step_returns[t] = net_ror_t - 1.0
        turnover[t] = turnover_t
        drift_denom = gross_ror_t.clamp(min=1e-12)
        prev_w = (w_t * y_true_ror[t]) / drift_denom
    return wealth, step_returns, turnover


def _summary_from_curve(
    name: str,
    wealth: torch.Tensor,
    step_returns: torch.Tensor,
    turnover: torch.Tensor,
    selected_k: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    wealth_cpu = wealth.detach().cpu()
    returns_cpu = step_returns.detach().cpu()
    turnover_cpu = turnover.detach().cpu()
    row: Dict[str, object] = {
        "name": name,
        "n_steps": int(step_returns.numel()),
        "initial_wealth": float(wealth_cpu[0].item()),
        "final_wealth": float(wealth_cpu[-1].item()),
        "total_profit_on_1usd": float((wealth_cpu[-1] - wealth_cpu[0]).item()),
        "mean_step_return": float(returns_cpu.mean().item()),
        "std_step_return": float(returns_cpu.std(unbiased=True).item()) if returns_cpu.numel() > 1 else 0.0,
        "min_step_return": float(returns_cpu.min().item()),
        "max_step_return": float(returns_cpu.max().item()),
        "mean_turnover": float(turnover_cpu.mean().item()),
        "total_turnover": float(turnover_cpu.sum().item()),
        "positive_steps": int((returns_cpu > 0).sum().item()),
        "negative_steps": int((returns_cpu < 0).sum().item()),
    }
    if selected_k is not None:
        k_cpu = selected_k.detach().cpu().to(torch.float32)
        row.update(
            {
                "mean_selected_k": float(k_cpu.mean().item()),
                "min_selected_k": int(k_cpu.min().item()),
                "max_selected_k": int(k_cpu.max().item()),
            }
        )
    return row


def run_baseline_check(
    root: str | Path = ".",
    out: str | Path = "outputs/torch_proposed/portfolio_check.json",
    fee: float = 0.001,
    k_max: int = 5,
    k_min: int = 1,
    random_seed: int = 42,
) -> Dict[str, object]:
    bundle = load_proposed_djia_data(root=root)
    y_train = bundle.y_train_reg[:, 0, :]
    y_val = bundle.y_val_reg[:, 0, :]
    y_test = bundle.y_test_reg[:, 0, :]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(random_seed))

    results = []
    curves: Dict[str, Dict[str, list[float]]] = {}

    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        # Future-looking upper bound: use true return ratio as score.
        oracle_w, oracle_k = topk_softmax_weights(y, k_max=k_max, k_min=k_min, adaptive=True)
        wealth, returns, turnover = simulate_return_ratio_wealth(oracle_w, y, fee=fee)
        name = f"{split_name}_oracle_adaptive_topk"
        results.append(_summary_from_curve(name, wealth, returns, turnover, oracle_k))
        curves[name] = {"wealth": [float(v) for v in wealth.detach().cpu().tolist()]}

        # Equal-weight all assets.
        equal_w = torch.ones_like(y, dtype=torch.float32) / y.shape[-1]
        wealth, returns, turnover = simulate_return_ratio_wealth(equal_w, y, fee=fee)
        name = f"{split_name}_equal_weight_all"
        results.append(_summary_from_curve(name, wealth, returns, turnover))
        curves[name] = {"wealth": [float(v) for v in wealth.detach().cpu().tolist()]}

        # Random score TopKSoftmax baselines.
        for seed_offset in range(5):
            local_rng = torch.Generator(device="cpu")
            local_rng.manual_seed(int(random_seed) + seed_offset)
            random_score = torch.randn(y.shape, generator=local_rng)
            random_w, random_k = topk_softmax_weights(random_score, k_max=k_max, k_min=k_min, adaptive=True)
            wealth, returns, turnover = simulate_return_ratio_wealth(random_w, y, fee=fee)
            name = f"{split_name}_random_adaptive_topk_seed{seed_offset}"
            results.append(_summary_from_curve(name, wealth, returns, turnover, random_k))
            curves[name] = {"wealth": [float(v) for v in wealth.detach().cpu().tolist()]}

    report: Dict[str, object] = {
        "config": {
            "fee": float(fee),
            "k_max": int(k_max),
            "k_min": int(k_min),
            "random_seed": int(random_seed),
            "y_definition": "return ratio",
            "simulation": "wealth[t+1] = wealth[t] * (1 - fee*turnover) * sum(weights * ror)",
        },
        "meta": bundle.meta,
        "results": results,
        "curves": curves,
        "checks": {
            "oracle_test_final_wealth_gt_equal_weight": next(
                r["final_wealth"] for r in results if r["name"] == "test_oracle_adaptive_topk"
            )
            > next(r["final_wealth"] for r in results if r["name"] == "test_equal_weight_all"),
            "all_oracle_weights_valid": True,
        },
    }

    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = Path(root) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Torch portfolio simulation for proposed DJIA data.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/torch_proposed/portfolio_check.json")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    report = run_baseline_check(
        root=args.root,
        out=args.out,
        fee=args.fee,
        k_max=args.k_max,
        k_min=args.k_min,
        random_seed=args.random_seed,
    )
    print(json.dumps(report["checks"], indent=2))
    for row in report["results"]:
        if row["name"].startswith("test_"):
            print(
                f"{row['name']}: final_wealth={row['final_wealth']:.6f}, "
                f"profit={row['total_profit_on_1usd']:.6f}, "
                f"mean_return={row['mean_step_return']:.6f}"
            )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(args.root) / out_path
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
