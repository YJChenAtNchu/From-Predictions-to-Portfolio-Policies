from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def calculate_portfolio_metrics(
    wealth_curve: torch.Tensor | np.ndarray,
    trading_freq: float = 15.0,
    mar: float = 0.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute APR/AVol/ASR/SoR/MDD/CR using the proposed TF metric convention.

    The original TensorFlow code uses Ny = 252 / trading_freq. For the cached
    DJIA experiment, trading_freq=15 means one portfolio decision every 15 days.
    """
    if isinstance(wealth_curve, torch.Tensor):
        wealth = wealth_curve.detach().cpu().numpy().astype(np.float64)
    else:
        wealth = np.asarray(wealth_curve, dtype=np.float64)

    wealth = np.squeeze(wealth)
    if wealth.ndim != 1:
        raise ValueError(f"Expected 1D wealth curve, got shape {wealth.shape}")
    if wealth.shape[0] < 2:
        raise ValueError("wealth_curve must contain at least two points")

    trade_ror = wealth[1:] / np.maximum(wealth[:-1], eps) - 1.0
    ny = 252.0 / float(trading_freq)
    at = float(np.mean(trade_ror))
    vt = float(np.std(trade_ror, ddof=1)) if trade_ror.size > 1 else 0.0

    apr = at * ny
    avol = vt * np.sqrt(ny)
    asr = apr / avol if abs(avol) > eps else 0.0

    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / np.maximum(running_max, eps)
    mdd = float(np.max(drawdown))
    cr = apr / mdd if abs(mdd) > eps else 0.0

    downside = np.clip(float(mar) - trade_ror, 0.0, np.inf)
    if np.any(downside > 0):
        downside_deviation = float(np.sqrt(np.mean(downside[downside > 0] ** 2)))
    else:
        downside_deviation = 0.0
    sor = apr / downside_deviation if abs(downside_deviation) > eps else 0.0

    return {
        "APV": float(wealth[-1] / max(wealth[0], eps)),
        "APR": float(apr),
        "AVol": float(avol),
        "ASR": float(asr),
        "SoR": float(sor),
        "MDD": float(mdd),
        "CR": float(cr),
    }
