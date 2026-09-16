from __future__ import annotations

import torch

from torch_portfolio_accounting_v2 import simulate_portfolio_accounting


def test_drift_aware_turnover_and_fee() -> None:
    weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    relatives = torch.tensor([[1.1, 0.9], [1.0, 1.0]], dtype=torch.float64)
    result = simulate_portfolio_accounting(weights, relatives, fee=0.001)
    torch.testing.assert_close(
        result.drift_weights[1], torch.tensor([0.55, 0.45], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.turnover_full_l1, torch.tensor([1.0, 0.10], dtype=torch.float64)
    )
    torch.testing.assert_close(result.net_relative[1], torch.tensor(0.9999, dtype=torch.float64))


def test_no_terminal_liquidation_fee() -> None:
    weights = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    relatives = torch.tensor([[1.10, 0.90]], dtype=torch.float64)
    result = simulate_portfolio_accounting(weights, relatives, fee=0.001)
    assert result.turnover_full_l1.shape == (1,)
    torch.testing.assert_close(result.turnover_full_l1[0], torch.tensor(1.0, dtype=torch.float64))
