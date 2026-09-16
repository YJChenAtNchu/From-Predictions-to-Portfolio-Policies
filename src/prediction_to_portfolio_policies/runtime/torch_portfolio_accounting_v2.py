from __future__ import annotations

from dataclasses import dataclass

import torch


ACCOUNTING_PROTOCOL_ID = "drift_full_l1_entry_fee_mark_to_market_v1_20260905"


@dataclass(frozen=True)
class PortfolioAccountingResult:
    wealth: torch.Tensor
    target_weights: torch.Tensor
    drift_weights: torch.Tensor
    gross_relative: torch.Tensor
    turnover_full_l1: torch.Tensor
    transaction_cost_rate: torch.Tensor
    net_relative: torch.Tensor
    net_returns: torch.Tensor


def _as_policy_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError(f"{name} supports only a singleton output horizon, got {tuple(value.shape)}")
        value = value[:, 0, :]
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (N,A) or (N,1,A), got {tuple(value.shape)}")
    return value.to(dtype=torch.float64)


def simulate_portfolio_accounting(
    target_weights: torch.Tensor,
    delayed_gross_relatives: torch.Tensor,
    fee: float = 0.001,
    initial_wealth: float = 1.0,
    weight_tolerance: float = 1e-6,
) -> PortfolioAccountingResult:
    """Canonical long-only accounting with full-L1 drift-aware turnover.

    The initial risky-asset holding is zero (cash before entry), so the first
    rebalance pays entry turnover. The final value is marked to market after
    the last holding interval; no terminal liquidation trade or fee is added.
    """

    weights = _as_policy_matrix(target_weights, "target_weights")
    relatives = _as_policy_matrix(delayed_gross_relatives, "delayed_gross_relatives")
    if weights.shape != relatives.shape:
        raise ValueError(f"weights/relatives shape mismatch: {tuple(weights.shape)} vs {tuple(relatives.shape)}")
    if not torch.isfinite(weights).all() or not torch.isfinite(relatives).all():
        raise ValueError("weights and relatives must be finite")
    if bool((relatives <= 0).any()):
        raise ValueError("gross price relatives must be strictly positive")
    if bool((weights < -weight_tolerance).any()):
        raise ValueError("canonical evaluator accepts long-only policies only")
    row_sums = weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=weight_tolerance):
        raise ValueError("canonical evaluator requires fully invested policy rows")
    if fee < 0:
        raise ValueError("fee must be non-negative")
    if initial_wealth <= 0:
        raise ValueError("initial wealth must be positive")

    n_steps, n_assets = weights.shape
    device = weights.device
    wealth = torch.empty(n_steps + 1, dtype=torch.float64, device=device)
    drift = torch.empty_like(weights)
    gross = torch.empty(n_steps, dtype=torch.float64, device=device)
    turnover = torch.empty(n_steps, dtype=torch.float64, device=device)
    cost_rate = torch.empty(n_steps, dtype=torch.float64, device=device)
    net = torch.empty(n_steps, dtype=torch.float64, device=device)
    wealth[0] = float(initial_wealth)

    pretrade = torch.zeros(n_assets, dtype=torch.float64, device=device)
    for step in range(n_steps):
        drift[step] = pretrade
        turnover[step] = torch.sum(torch.abs(weights[step] - pretrade))
        gross[step] = torch.sum(weights[step] * relatives[step])
        cost_rate[step] = float(fee) * turnover[step]
        if bool(cost_rate[step] >= 1.0):
            raise ValueError("transaction cost rate must remain below one")
        net[step] = (1.0 - cost_rate[step]) * gross[step]
        wealth[step + 1] = wealth[step] * net[step]
        pretrade = (weights[step] * relatives[step]) / gross[step]

    return PortfolioAccountingResult(
        wealth=wealth,
        target_weights=weights,
        drift_weights=drift,
        gross_relative=gross,
        turnover_full_l1=turnover,
        transaction_cost_rate=cost_rate,
        net_relative=net,
        net_returns=net - 1.0,
    )
