from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
from torch import nn


class EMALossNormalizer:
    """EMA-based loss-scale normalizer, matching the TensorFlow plan.

    The TensorFlow proposed code normalizes each loss component by a running
    exponential moving average.  We keep the same idea here while using the
    magnitude of each component as the denominator so losses such as negative
    Sharpe remain numerically stable.
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._ema: Dict[str, torch.Tensor] = {}
        self._updates_enabled = True

    def denominator(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        value = loss.detach().mean().abs()
        if name not in self._ema:
            self._ema[name] = value.clone()
        elif self._updates_enabled:
            prev = self._ema[name].to(value.device)
            self._ema[name] = ((1.0 - self.alpha) * prev + self.alpha * value).detach()
        return self._ema[name].to(loss.device).clamp(min=self.eps)

    @contextmanager
    def frozen(self):
        """Read existing EMA scales without validation data updating them."""
        previous = self._updates_enabled
        self._updates_enabled = False
        try:
            yield self
        finally:
            self._updates_enabled = previous

    def normalize(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        return loss / self.denominator(name, loss)

    def snapshot(self) -> Dict[str, float]:
        return {key: float(value.detach().cpu().item()) for key, value in self._ema.items()}


def pairwise_ranking_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    ema_normalizer: EMALossNormalizer | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Pure pairwise cross-sectional ranking loss for regression outputs.

    The ranking term is computed cross-sectionally across assets inside each
    sample and each output time step.  This matches the slide formula:

        mean_{b,t,a!=a'} relu(-(yhat_a - yhat_a') * (y_a - y_a'))

    We intentionally do not flatten batch/time/assets before ranking, because
    that would compare assets from different samples or dates.  The cached DJIA
    target is a gross return ratio, so callers should pass centered return rate
    (y_reg - 1) when using this objective.  Pairwise ordering is unchanged by
    subtracting one.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"pairwise_ranking_loss shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}")
    if y_pred.ndim != 3:
        raise ValueError(f"pairwise_ranking_loss expects shape (B,T,A), got {tuple(y_pred.shape)}")
    n_assets = int(y_pred.shape[-1])
    if n_assets < 2:
        raise ValueError("pairwise_ranking_loss needs at least two assets for pairwise ranking")

    pred_diff = y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2)
    true_diff = y_true.unsqueeze(-1) - y_true.unsqueeze(-2)
    pair_loss = torch.relu(-(pred_diff * true_diff))

    # Exclude diagonal asset pairs a == a'; they carry no ranking information.
    off_diag = ~torch.eye(n_assets, dtype=torch.bool, device=y_pred.device)
    rank = pair_loss[..., off_diag].mean()
    rank_term = ema_normalizer.normalize("ranking", rank) if ema_normalizer is not None else rank
    return rank_term, {
        "ranking": float(rank.detach().cpu().item()),
    }


def directional_product_ranking_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    ema_normalizer: EMALossNormalizer | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Legacy direction-consistency ranking surrogate.

    This is the earlier product-based formulation used before switching to the
    paper-style pairwise-difference ranking loss.  The term is still computed
    cross-sectionally inside each sample/time step, but it checks whether pairs
    of assets have compatible *directions* rather than whether their relative
    ordering is correct:

        mean_{b,t,a!=a'} relu(-(yhat_a * yhat_a') * (y_a * y_a'))

    Callers should pass centered return rates, i.e. ``reg_out - 1`` and
    ``y_reg - 1`` for gross return ratio data.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"directional_product_ranking_loss shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}"
        )
    if y_pred.ndim != 3:
        raise ValueError(f"directional_product_ranking_loss expects shape (B,T,A), got {tuple(y_pred.shape)}")
    n_assets = int(y_pred.shape[-1])
    if n_assets < 2:
        raise ValueError("directional_product_ranking_loss needs at least two assets")

    pred_prod = y_pred.unsqueeze(-1) * y_pred.unsqueeze(-2)
    true_prod = y_true.unsqueeze(-1) * y_true.unsqueeze(-2)
    product_loss = torch.relu(-(pred_prod * true_prod))

    off_diag = ~torch.eye(n_assets, dtype=torch.bool, device=y_pred.device)
    rank = product_loss[..., off_diag].mean()
    rank_term = ema_normalizer.normalize("product_ranking", rank) if ema_normalizer is not None else rank
    return rank_term, {
        "product_ranking": float(rank.detach().cpu().item()),
    }


def ranking_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    rank_weight: float = 0.5,
    ema_normalizer: EMALossNormalizer | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Backward-compatible ranking+MSE helper.

    New training code computes MSE and ranking as separate loss terms because
    PyTorch does not have the Keras compile constraint that motivated bundling
    them together.  This wrapper is kept for older scripts/tests that may still
    import ``ranking_mse_loss`` directly.
    """
    rank_term, rank_logs = pairwise_ranking_loss(y_pred, y_true, ema_normalizer=ema_normalizer)
    mse = nn.functional.mse_loss(y_pred, y_true)
    mse_term = ema_normalizer.normalize("ranking_mse", mse) if ema_normalizer is not None else mse
    loss = float(rank_weight) * rank_term + (1.0 - float(rank_weight)) * mse_term
    return loss, {
        **rank_logs,
        "mse": float(mse.detach().cpu().item()),
        "ranking_mse": float(loss.detach().cpu().item()),
    }


def contrastive_loss(con_out: torch.Tensor, margin: float = 1.0) -> tuple[torch.Tensor, Dict[str, float]]:
    """Vectorized contrastive loss matching proposed/loss.py.

    The TensorFlow version reshapes the representation to (N, D), computes all
    pairwise squared distances, masks diagonal pairs, and averages
    relu(margin - distance).
    """
    z = con_out.reshape(con_out.shape[0], -1)
    diff = z[:, None, :] - z[None, :, :]
    dist_sq = diff.square().sum(dim=-1)
    mask = 1.0 - torch.eye(z.shape[0], dtype=z.dtype, device=z.device)
    loss_mat = torch.relu(float(margin) - dist_sq) * mask
    loss = loss_mat.mean()
    return loss, {
        "contrastive": float(loss.detach().cpu().item()),
    }


def sharpe_loss(
    pred_rate: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable negative Sharpe-style penalty on predicted return rates.

    The TensorFlow source currently returns mean_return / mean_return, which is
    effectively constant. This Torch version uses a trainable approximation:
    loss = - mean(predicted returns) / std(predicted returns).
    """
    flat = pred_rate.reshape(-1)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp(min=eps)
    sharpe = mean / std
    loss = -sharpe
    return loss, {
        "sharpe_loss": float(loss.detach().cpu().item()),
        "pred_sharpe": float(sharpe.detach().cpu().item()),
    }
