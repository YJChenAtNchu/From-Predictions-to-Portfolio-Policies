from __future__ import annotations

from typing import Dict, List

import torch

from torch_metrics import calculate_portfolio_metrics
from torch_portfolio import simulate_return_ratio_wealth, topk_softmax_weights


@torch.no_grad()
def predict_weak_learner(
    model: torch.nn.Module,
    x_list: List[torch.Tensor],
    *,
    k_min: int,
    k_max: int,
    decision_rule: str,
    eta_k: float,
    eta_omega: float,
) -> Dict[str, torch.Tensor]:
    model.eval()
    outputs = model(x_list)
    policy, selected_k = topk_softmax_weights(
        outputs["reg_out"],
        k_min=k_min,
        k_max=k_max,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
    )
    return {
        "reg_out": outputs["reg_out"],
        "cls_prob": torch.softmax(outputs["cls_logits"], dim=-1),
        "policy": policy,
        "selected_k": selected_k,
    }


def update_weighted_ensemble(
    accumulator: Dict[str, torch.Tensor] | None,
    prediction: Dict[str, torch.Tensor],
    alpha_weight: float,
) -> Dict[str, torch.Tensor]:
    weight = max(float(alpha_weight), 0.0)
    if accumulator is None:
        accumulator = {
            "policy_sum": torch.zeros_like(prediction["policy"]),
            "cls_prob_sum": torch.zeros_like(prediction["cls_prob"]),
            "reg_out_sum": torch.zeros_like(prediction["reg_out"]),
            "alpha_total": torch.zeros(
                (), dtype=prediction["policy"].dtype, device=prediction["policy"].device
            ),
            "fallback_policy_sum": torch.zeros_like(prediction["policy"]),
            "fallback_cls_prob_sum": torch.zeros_like(prediction["cls_prob"]),
            "fallback_reg_out_sum": torch.zeros_like(prediction["reg_out"]),
            "learner_count": torch.zeros(
                (), dtype=prediction["policy"].dtype, device=prediction["policy"].device
            ),
        }
    accumulator["fallback_policy_sum"].add_(prediction["policy"])
    accumulator["fallback_cls_prob_sum"].add_(prediction["cls_prob"])
    accumulator["fallback_reg_out_sum"].add_(prediction["reg_out"])
    accumulator["learner_count"].add_(1.0)
    if weight > 0.0:
        accumulator["policy_sum"].add_(prediction["policy"], alpha=weight)
        accumulator["cls_prob_sum"].add_(prediction["cls_prob"], alpha=weight)
        accumulator["reg_out_sum"].add_(prediction["reg_out"], alpha=weight)
        accumulator["alpha_total"].add_(weight)
    return accumulator


def materialize_weighted_ensemble(
    accumulator: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    alpha_total = accumulator["alpha_total"]
    if float(alpha_total.detach().cpu().item()) <= 1e-12:
        learner_count = accumulator["learner_count"].clamp(min=1.0)
        return {
            "policy": accumulator["fallback_policy_sum"] / learner_count,
            "cls_prob": accumulator["fallback_cls_prob_sum"] / learner_count,
            "reg_out": accumulator["fallback_reg_out_sum"] / learner_count,
            "alpha_total": alpha_total,
            "used_fallback": torch.ones((), dtype=torch.bool, device=alpha_total.device),
        }
    return {
        "policy": accumulator["policy_sum"] / alpha_total,
        "cls_prob": accumulator["cls_prob_sum"] / alpha_total,
        "reg_out": accumulator["reg_out_sum"] / alpha_total,
        "alpha_total": alpha_total,
        "used_fallback": torch.zeros((), dtype=torch.bool, device=alpha_total.device),
    }


@torch.no_grad()
def evaluate_policy_ensemble(
    ensemble: Dict[str, torch.Tensor],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    *,
    fee: float,
    trading_freq: float,
) -> Dict[str, object]:
    policy = ensemble["policy"]
    cls_prob = ensemble["cls_prob"]
    target = y_cls.argmax(dim=-1)
    prediction = cls_prob.argmax(dim=-1)
    acc = float((prediction == target).to(torch.float32).mean().mul(100.0).item())

    wealth, step_returns, turnover = simulate_return_ratio_wealth(
        policy,
        y_reg,
        fee=fee,
    )
    financial_metrics = calculate_portfolio_metrics(
        wealth,
        trading_freq=trading_freq,
    )
    y_ror = y_reg[:, 0, :] if y_reg.ndim == 3 else y_reg
    sample_profit = torch.sum(policy * y_ror, dim=-1) - 1.0
    support_size = (policy > 1e-12).sum(dim=-1).to(torch.float32)
    concentration = policy.square().sum(dim=-1)
    return {
        "acc": acc,
        "sum_one_step_profit": float(sample_profit.sum().item()),
        "mean_one_step_profit": float(sample_profit.mean().item()),
        "final_wealth": float(wealth[-1].item()),
        "profit_on_1usd": float((wealth[-1] - wealth[0]).item()),
        "mean_step_return": float(step_returns.mean().item()),
        "total_turnover": float(turnover.sum().item()),
        "mean_support_size": float(support_size.mean().item()),
        "std_support_size": (
            float(support_size.std(unbiased=True).item()) if support_size.numel() > 1 else 0.0
        ),
        "mean_policy_hhi": float(concentration.mean().item()),
        "alpha_total": float(ensemble["alpha_total"].item()),
        "used_zero_alpha_fallback": bool(ensemble["used_fallback"].item()),
        "financial_metrics": financial_metrics,
        "_policy": policy,
        "_cls_prob": cls_prob,
        "_reg_out": ensemble["reg_out"],
        "_wealth": wealth,
        "_turnover": turnover,
    }
