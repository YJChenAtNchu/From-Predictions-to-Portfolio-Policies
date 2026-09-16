from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from torch_data import ProposedDataBundle, load_proposed_djia_data
from torch_metrics import calculate_portfolio_metrics
from torch_model import build_model_from_data
from torch_portfolio import simulate_return_ratio_wealth, topk_softmax_weights
from torch_losses import (
    EMALossNormalizer,
    contrastive_loss,
    directional_product_ranking_loss,
    pairwise_ranking_loss,
    ranking_mse_loss,
    sharpe_loss,
)
from torch_train_smoke import _batch_indices, _take_batch, classification_accuracy, set_seed
from experiment_snapshot import save_code_snapshot


def get_w0_from_rate(y_rate: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Initial sample weights, following exp(z_i^2 / sum_j z_j^2).

    The cached DJIA y_reg is return ratio, so we use y_rate = y_reg - 1 as z.
    """
    if y_rate.ndim != 3:
        raise ValueError(f"Expected y_rate shape (N,1,A), got {tuple(y_rate.shape)}")
    z = y_rate.sum(dim=(1, 2))
    z2 = z.square()
    return torch.exp(z2 / z2.sum().clamp(min=eps))


def rescale_profit(x: torch.Tensor, r_max: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map profit from [-r_max, r_max] to [1, 0], matching the TF formula."""
    tau = (x - r_max) / ((-r_max) - r_max).clamp(max=-eps)
    return torch.clamp(tau, 1e-4, 1.0 - 1e-4)


def weighted_ce_mse_loss(
    outputs: Dict[str, torch.Tensor],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    sample_weight: torch.Tensor,
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
    loss_mode: str = "ce_mse",
    rank_weight: float = 0.5,
    use_contrastive: bool = False,
    contrastive_weight: float = 1.0,
    contrastive_margin: float = 1.0,
    use_sharpe: bool = False,
    sharpe_weight: float = 1.0,
    ema_normalizer: EMALossNormalizer | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls_logits = outputs["cls_logits"]
    cls_target = y_cls.argmax(dim=-1).to(torch.long)

    ce_per_item = nn.functional.cross_entropy(
        cls_logits.reshape(-1, 2),
        cls_target.reshape(-1),
        reduction="none",
    ).reshape(cls_target.shape)
    ce_per_sample = ce_per_item.mean(dim=(1, 2))
    mse_per_sample = (outputs["reg_out"] - y_reg).square().mean(dim=(1, 2))
    ce_term = ce_per_sample
    mse_term = mse_per_sample
    if ema_normalizer is not None:
        ce_term = ce_per_sample / ema_normalizer.denominator("ce", ce_per_sample.mean())
        mse_term = mse_per_sample / ema_normalizer.denominator("mse", mse_per_sample.mean())
    if loss_mode == "ce_only":
        sample_loss = ce_weight * ce_term
        reg_logs = {"mse": float(mse_per_sample.mean().detach().cpu().item())}
        ranking_term = None
    elif loss_mode == "ce_mse":
        sample_loss = ce_weight * ce_term + mse_weight * mse_term
        reg_logs = {"mse": float(mse_per_sample.mean().detach().cpu().item())}
        ranking_term = None
    elif loss_mode in {"ce_rank", "ranking_mse", "ce_mse_rank", "ce_product_rank", "product_ranking_mse", "ce_mse_product_rank"}:
        # MSE and ranking both use reg_out/y_reg, but they are independent
        # terms in Torch.  Ranking operates on centered return rates to avoid
        # ranking all gross return ratios around +1.
        use_product_rank = loss_mode in {"ce_product_rank", "product_ranking_mse", "ce_mse_product_rank"}
        rank_fn = directional_product_ranking_loss if use_product_rank else pairwise_ranking_loss
        rank_loss, rank_logs = rank_fn(outputs["reg_out"] - 1.0, y_reg - 1.0, ema_normalizer=ema_normalizer)
        if loss_mode in {"ce_rank", "ce_product_rank"}:
            sample_loss = ce_weight * ce_term
        else:
            sample_loss = ce_weight * ce_term + mse_weight * mse_term
        # Keep sample-specific weights on CE/MSE, and add ranking as a
        # batch-level cross-sectional differentiable term.
        ranking_term = float(rank_weight) * rank_loss
        reg_logs = {
            "mse": float(mse_per_sample.mean().detach().cpu().item()),
            "ranking_weighted": float(ranking_term.detach().cpu().item()),
            **rank_logs,
        }
    elif loss_mode == "ranking_mse_wrapper":
        # Legacy wrapper mode used by earlier experiments: CE/MSE are computed
        # as sample-weighted terms, and the regression wrapper adds another
        # batch-level 0.5*ranking + 0.5*MSE surrogate.  Keep this mode explicit
        # so it can be compared against the cleaner split implementation.
        rank_loss, rank_logs = ranking_mse_loss(
            outputs["reg_out"] - 1.0,
            y_reg - 1.0,
            rank_weight=rank_weight,
            ema_normalizer=ema_normalizer,
        )
        sample_loss = ce_weight * ce_term + mse_weight * mse_term
        ranking_term = rank_loss
        reg_logs = rank_logs
    else:
        raise ValueError(f"Unsupported loss_mode={loss_mode}")

    # Normalize within mini-batch for stable gradients while preserving relative weights.
    w = sample_weight / sample_weight.mean().clamp(min=1e-12)
    loss = (sample_loss * w).mean()
    if ranking_term is not None:
        loss = loss + ranking_term
    if use_contrastive:
        con_loss, con_logs = contrastive_loss(outputs["con_out"], margin=contrastive_margin)
        if ema_normalizer is not None:
            con_loss = ema_normalizer.normalize("contrastive", con_loss)
        loss = loss + float(contrastive_weight) * con_loss
        reg_logs.update(con_logs)
    if use_sharpe:
        shp_loss, shp_logs = sharpe_loss(outputs["reg_out"] - 1.0)
        if ema_normalizer is not None:
            shp_loss = ema_normalizer.normalize("sharpe", shp_loss)
        loss = loss + float(sharpe_weight) * shp_loss
        reg_logs.update(shp_logs)
    if ema_normalizer is not None:
        reg_logs["ema_loss_norm"] = 1.0
    reg_logs["total_with_extra"] = float(loss.detach().cpu().item())
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "ce": float(ce_per_sample.mean().detach().cpu().item()),
        "sample_loss_mean": float(sample_loss.mean().detach().cpu().item()),
        **reg_logs,
    }


@torch.no_grad()
def one_step_portfolio_profit(
    pred_reg: torch.Tensor,
    y_true_ror: torch.Tensor,
    k_max: int,
    k_min: int,
    decision_rule: str = "spread",
    eta_k: float = 1.0,
    eta_omega: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sample-wise one-step gross profit and allocation weights."""
    weights, _ = topk_softmax_weights(
        pred_reg,
        k_max=k_max,
        k_min=k_min,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
    )
    if y_true_ror.ndim == 3:
        y_true_ror = y_true_ror[:, 0, :]
    profit_pre = torch.sum(weights * y_true_ror, dim=-1) - 1.0
    return profit_pre, weights


@torch.no_grad()
def portfolio_path_profit(
    pred_reg: torch.Tensor,
    y_true_ror: torch.Tensor,
    k_max: int,
    k_min: int,
    fee: float,
    decision_rule: str = "spread",
    eta_k: float = 1.0,
    eta_omega: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TF-style path profit from a wealth simulator.

    This mirrors the legacy TensorFlow PortfolioSim usage: weights are
    generated for the whole path, transaction costs are applied through the
    simulator, and the returned profit is final wealth minus initial wealth.
    """
    weights, _ = topk_softmax_weights(
        pred_reg,
        k_max=k_max,
        k_min=k_min,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
    )
    wealth, _, turnover = simulate_return_ratio_wealth(weights, y_true_ror, fee=fee)
    return wealth[-1] - wealth[0], weights, turnover


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    x_list: List[torch.Tensor],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    fee: float,
    k_max: int,
    k_min: int,
    trading_freq: float,
    decision_rule: str = "spread",
    eta_k: float = 1.0,
    eta_omega: float = 1.0,
    return_outputs: bool = False,
) -> Dict[str, object]:
    model.eval()
    outputs = model(x_list)
    acc = classification_accuracy(outputs["cls_logits"], y_cls)
    weights, selected_k, decision_diagnostics = topk_softmax_weights(
        outputs["reg_out"],
        k_max=k_max,
        k_min=k_min,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
        return_diagnostics=True,
    )
    wealth, step_returns, turnover = simulate_return_ratio_wealth(weights, y_reg, fee=fee)
    profit_pre, _ = one_step_portfolio_profit(
        outputs["reg_out"],
        y_reg,
        k_max=k_max,
        k_min=k_min,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
    )
    fin_metrics = calculate_portfolio_metrics(wealth, trading_freq=trading_freq)
    result: Dict[str, object] = {
        "acc": acc,
        "sum_one_step_profit": float(profit_pre.sum().detach().cpu().item()),
        "mean_one_step_profit": float(profit_pre.mean().detach().cpu().item()),
        "final_wealth": float(wealth[-1].detach().cpu().item()),
        "profit_on_1usd": float((wealth[-1] - wealth[0]).detach().cpu().item()),
        "mean_step_return": float(step_returns.detach().cpu().mean().item()),
        "total_turnover": float(turnover.detach().cpu().sum().item()),
        "financial_metrics": fin_metrics,
        **_selection_diagnostics(
            selected_k,
            decision_diagnostics,
            k_min=k_min,
            k_max=k_max,
        ),
    }
    if return_outputs:
        result["_outputs"] = outputs
        result["_weights"] = weights
        result["_selected_k"] = selected_k
        result["_decision_diagnostics"] = decision_diagnostics
        result["_wealth"] = wealth
    return result


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x_cpu = x.detach().cpu().to(torch.float32)
    return {
        "min": float(x_cpu.min().item()),
        "mean": float(x_cpu.mean().item()),
        "max": float(x_cpu.max().item()),
        "std": float(x_cpu.std(unbiased=True).item()) if x_cpu.numel() > 1 else 0.0,
    }


def train_one_outer_round(
    model: nn.Module,
    bundle: ProposedDataBundle,
    sample_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    inner_epochs: int,
    inner_patience: int,
    ce_weight: float,
    mse_weight: float,
    loss_mode: str,
    rank_weight: float,
    use_contrastive: bool,
    contrastive_weight: float,
    contrastive_margin: float,
    use_sharpe: bool,
    sharpe_weight: float,
    use_ema_loss_norm: bool,
    ema_alpha: float,
    seed: int,
    outer_round: int,
    log_every_epoch: int,
    device: str,
) -> Dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1000 + outer_round)
    n_train = int(bundle.y_train_reg.shape[0])
    epoch_losses = []
    best_val_loss = float("inf")
    best_state = None
    inner_bad = 0
    epochs_ran = 0
    ema_normalizer = EMALossNormalizer(alpha=ema_alpha) if use_ema_loss_norm else None
    for epoch_idx in range(inner_epochs):
        model.train()
        epoch_loss_rows = []
        for idx_cpu in _batch_indices(n_train, batch_size, generator):
            idx = idx_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(_take_batch(bundle.x_train_list, idx))
            loss, losses = weighted_ce_mse_loss(
                outputs,
                bundle.y_train_reg[idx],
                bundle.y_train_cls[idx],
                sample_weight[idx],
                ce_weight=ce_weight,
                mse_weight=mse_weight,
                loss_mode=loss_mode,
                rank_weight=rank_weight,
                use_contrastive=use_contrastive,
                contrastive_weight=contrastive_weight,
                contrastive_margin=contrastive_margin,
                use_sharpe=use_sharpe,
                sharpe_weight=sharpe_weight,
                ema_normalizer=ema_normalizer,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(losses)
            epoch_loss_rows.append(losses)
        epochs_ran = epoch_idx + 1
        val_loss = None
        if inner_patience > 0:
            model.eval()
            val_rows = []
            ema_context = ema_normalizer.frozen() if ema_normalizer is not None else nullcontext()
            with torch.no_grad(), ema_context:
                n_val = int(bundle.y_val_reg.shape[0])
                for start in range(0, n_val, batch_size):
                    idx = torch.arange(start, min(start + batch_size, n_val), device=device)
                    val_outputs = model(_take_batch(bundle.x_val_list, idx))
                    _, val_losses = weighted_ce_mse_loss(
                        val_outputs,
                        bundle.y_val_reg[idx],
                        bundle.y_val_cls[idx],
                        torch.ones(idx.numel(), dtype=torch.float32, device=device),
                        ce_weight=ce_weight,
                        mse_weight=mse_weight,
                        loss_mode=loss_mode,
                        rank_weight=rank_weight,
                        use_contrastive=use_contrastive,
                        contrastive_weight=contrastive_weight,
                        contrastive_margin=contrastive_margin,
                        use_sharpe=use_sharpe,
                        sharpe_weight=sharpe_weight,
                        ema_normalizer=ema_normalizer,
                    )
                    val_rows.append(val_losses)
            val_loss = float(np.mean([row["loss"] for row in val_rows])) if val_rows else float("inf")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                inner_bad = 0
            else:
                inner_bad += 1
        if log_every_epoch > 0 and (
            epoch_idx == 0
            or (epoch_idx + 1) % log_every_epoch == 0
            or epoch_idx + 1 == inner_epochs
            or (inner_patience > 0 and inner_bad >= inner_patience)
        ):
            epoch_mean = {
                key: float(np.mean([row[key] for row in epoch_loss_rows]))
                for key in epoch_loss_rows[0]
            }
            val_msg = "" if val_loss is None else f" val_loss={val_loss:.6f} inner_bad={inner_bad}/{inner_patience}"
            print(
                f"outer_k {outer_round} epoch {epoch_idx + 1}/{inner_epochs} "
                f"loss={epoch_mean.get('loss', 0.0):.6f} "
                f"ce={epoch_mean.get('ce', 0.0):.6f} "
                f"mse={epoch_mean.get('mse', 0.0):.6f}"
                f"{val_msg}",
                flush=True,
            )
        if inner_patience > 0 and inner_bad >= inner_patience:
            print(
                f"outer_k {outer_round} inner early stop at epoch {epoch_idx + 1}: "
                f"best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {
        "loss": float(np.mean([row["loss"] for row in epoch_losses])),
        "ce": float(np.mean([row["ce"] for row in epoch_losses])),
        "mse": float(np.mean([row["mse"] for row in epoch_losses])),
        "epochs_ran": int(epochs_ran),
        "best_val_loss": float(best_val_loss) if inner_patience > 0 else None,
        "ema_loss_scale": ema_normalizer.snapshot() if ema_normalizer is not None else None,
    }


def _selection_diagnostics(
    selected_k: torch.Tensor,
    decision_diagnostics: Dict[str, torch.Tensor | str],
    k_min: int,
    k_max: int,
) -> Dict[str, float | int | bool | None]:
    k_cpu = selected_k.detach().cpu().to(torch.float32)
    counts = {
        k: int((selected_k.detach().cpu() == k).sum().item())
        for k in range(k_min, k_max + 1)
    }
    total = max(int(selected_k.numel()), 1)
    dominant_k = max(counts, key=counts.get)
    dominant_ratio = counts[dominant_k] / float(total)
    result: Dict[str, float | int | bool | None] = {
        "mean_selected_k": float(k_cpu.mean().item()),
        "std_selected_k": (
            float(k_cpu.std(unbiased=True).item()) if k_cpu.numel() > 1 else 0.0
        ),
        "min_selected_k": int(k_cpu.min().item()),
        "max_selected_k": int(k_cpu.max().item()),
        "dominant_k": int(dominant_k),
        "dominant_k_ratio": float(dominant_ratio),
        "is_degenerate": bool(dominant_ratio >= 0.90),
        "HHI_mean": None,
        "HHI_std": None,
        "N_eff_mean": None,
        "N_eff_std": None,
    }
    for k in range(k_min, k_max + 1):
        result[f"k{k}_ratio"] = counts[k] / float(total)

    hhi = decision_diagnostics.get("hhi")
    n_eff = decision_diagnostics.get("n_eff")
    if isinstance(hhi, torch.Tensor):
        hhi_cpu = hhi.detach().cpu().to(torch.float32)
        result["HHI_mean"] = float(hhi_cpu.mean().item())
        result["HHI_std"] = (
            float(hhi_cpu.std(unbiased=True).item()) if hhi_cpu.numel() > 1 else 0.0
        )
    if isinstance(n_eff, torch.Tensor):
        n_eff_cpu = n_eff.detach().cpu().to(torch.float32)
        result["N_eff_mean"] = float(n_eff_cpu.mean().item())
        result["N_eff_std"] = (
            float(n_eff_cpu.std(unbiased=True).item())
            if n_eff_cpu.numel() > 1
            else 0.0
        )
    return result


def train_profitboost(
    root: str | Path = ".",
    out_dir: str | Path = "outputs/torch_proposed/profitboost_ce_mse_seed001",
    seed: int = 1,
    max_outer_rounds: int = 20,
    inner_epochs: int = 5,
    inner_patience: int = 0,
    outer_patience: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    beta: float = 0.8,
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
    loss_mode: str = "ce_mse",
    rank_weight: float = 0.5,
    use_contrastive: bool = False,
    contrastive_weight: float = 1.0,
    contrastive_margin: float = 1.0,
    use_sharpe: bool = False,
    sharpe_weight: float = 1.0,
    use_ema_loss_norm: bool = False,
    ema_alpha: float = 0.1,
    fee: float = 0.001,
    k_max: int = 5,
    k_min: int = 1,
    decision_rule: str = "spread",
    eta_k: float = 1.0,
    eta_omega: float = 1.0,
    trading_freq: float = 15.0,
    frontend_type: str = "dense",
    lista_unroll_steps: int = 4,
    lbp_unroll_steps: int = 4,
    dropout_rate: float = 0.0,
    use_signature: bool = False,
    signature_level: int = 2,
    weight_update_mode: str = "sample_specific",
    profit_source: str = "one_step_sum",
    log_every_epoch: int = 0,
    device: str = "auto",
) -> Dict[str, object]:
    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root_path = Path(root)
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    snapshot_manifest = save_code_snapshot(
        out_path,
        root=root_path,
        extra_files=[Path(__file__)],
        note="offline ProfitBoost training",
    )

    bundle = load_proposed_djia_data(
        root=root_path,
        device=device,
        use_signature=use_signature,
        signature_level=signature_level,
    )
    model = build_model_from_data(
        bundle,
        frontend_type=frontend_type,
        lista_unroll_steps=lista_unroll_steps,
        lbp_unroll_steps=lbp_unroll_steps,
        dropout_rate=dropout_rate,
    ).to(device)
    model(_take_batch(bundle.x_train_list, torch.arange(min(batch_size, bundle.y_train_reg.shape[0]), device=device)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    w0 = get_w0_from_rate(bundle.y_train_rate).to(device)
    w = w0.clone()

    if profit_source == "one_step_sum":
        # A simple upper scale for tau: sum of oracle one-step profit magnitudes.
        oracle_profit, _ = one_step_portfolio_profit(
            bundle.y_train_reg,
            bundle.y_train_reg,
            k_max=k_max,
            k_min=k_min,
            decision_rule=decision_rule,
            eta_k=eta_k,
            eta_omega=eta_omega,
        )
        r_max = oracle_profit.abs().sum().clamp(min=1e-8)
    elif profit_source == "wealth_path":
        # TF-style scale: use an oracle path profit from the wealth simulator,
        # including compounding and transaction costs.
        oracle_path_profit, _, _ = portfolio_path_profit(
            bundle.y_train_reg,
            bundle.y_train_reg,
            k_max=k_max,
            k_min=k_min,
            fee=fee,
            decision_rule=decision_rule,
            eta_k=eta_k,
            eta_omega=eta_omega,
        )
        r_max = oracle_path_profit.abs().clamp(min=1e-8)
    else:
        raise ValueError(f"Unsupported profit_source={profit_source}")

    best_val_profit = -float("inf")
    best_state = None
    bad_rounds = 0
    outer_log = []

    for outer_round in range(max_outer_rounds):
        print(f"outer_k {outer_round} start", flush=True)
        w_cur = w / w.sum().clamp(min=1e-12)
        fit_log = train_one_outer_round(
            model=model,
            bundle=bundle,
            sample_weight=w_cur,
            optimizer=optimizer,
            batch_size=batch_size,
            inner_epochs=inner_epochs,
            inner_patience=inner_patience,
            ce_weight=ce_weight,
            mse_weight=mse_weight,
            loss_mode=loss_mode,
            rank_weight=rank_weight,
            use_contrastive=use_contrastive,
            contrastive_weight=contrastive_weight,
            contrastive_margin=contrastive_margin,
            use_sharpe=use_sharpe,
            sharpe_weight=sharpe_weight,
            use_ema_loss_norm=use_ema_loss_norm,
            ema_alpha=ema_alpha,
            seed=seed,
            outer_round=outer_round,
            log_every_epoch=log_every_epoch,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            train_outputs = model(bundle.x_train_list)
            profit_pre, _ = one_step_portfolio_profit(
                train_outputs["reg_out"],
                bundle.y_train_reg,
                k_max=k_max,
                k_min=k_min,
                decision_rule=decision_rule,
                eta_k=eta_k,
                eta_omega=eta_omega,
            )
            if profit_source == "one_step_sum":
                train_profit = profit_pre.sum()
            elif profit_source == "wealth_path":
                train_profit, _, _ = portfolio_path_profit(
                    train_outputs["reg_out"],
                    bundle.y_train_reg,
                    k_max=k_max,
                    k_min=k_min,
                    fee=fee,
                    decision_rule=decision_rule,
                    eta_k=eta_k,
                    eta_omega=eta_omega,
                )
            else:
                raise ValueError(f"Unsupported profit_source={profit_source}")
            tau = rescale_profit(train_profit, r_max)
            alpha = 0.5 * torch.log((1.0 - tau) / tau)
            alpha_weight = torch.clamp(alpha, min=0.0)
            if weight_update_mode == "sample_specific":
                a = torch.exp(-alpha_weight * torch.tanh(profit_pre))
                weight_update_desc = "W = beta*W0 + (1-beta)*(W*a), a=exp(-max(alpha,0)*tanh(sample_profit_pre))"
            elif weight_update_mode == "scalar_tf_style":
                # TensorFlow legacy behavior can collapse the portfolio profit
                # term to a scalar, making every sample weight scale together.
                a = torch.exp(-alpha_weight * torch.tanh(train_profit))
                weight_update_desc = f"W = beta*W0 + (1-beta)*(W*a), a=exp(-max(alpha,0)*tanh(total_train_profit)) scalar TF-style, profit_source={profit_source}"
            else:
                raise ValueError(f"Unsupported weight_update_mode={weight_update_mode}")

            val_metrics = evaluate_split(
                model,
                bundle.x_val_list,
                bundle.y_val_reg,
                bundle.y_val_cls,
                fee=fee,
                k_max=k_max,
                k_min=k_min,
                trading_freq=trading_freq,
                decision_rule=decision_rule,
                eta_k=eta_k,
                eta_omega=eta_omega,
            )
            train_metrics = evaluate_split(
                model,
                bundle.x_train_list,
                bundle.y_train_reg,
                bundle.y_train_cls,
                fee=fee,
                k_max=k_max,
                k_min=k_min,
                trading_freq=trading_freq,
                decision_rule=decision_rule,
                eta_k=eta_k,
                eta_omega=eta_omega,
            )

            val_profit = float(val_metrics["profit_on_1usd"])
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_rounds = 0
            else:
                bad_rounds += 1

            log_row = {
                "outer_round": outer_round,
                "fit_loss": fit_log,
                "train_profit": float(train_profit.detach().cpu().item()),
                "val_profit": val_profit,
                "tau": float(tau.detach().cpu().item()),
                "alpha": float(alpha.detach().cpu().item()),
                "alpha_weight": float(alpha_weight.detach().cpu().item()),
                "r_max": float(r_max.detach().cpu().item()),
                "W": _tensor_stats(w),
                "W_cur": _tensor_stats(w_cur),
                "profit_pre": {
                    "shape": list(profit_pre.shape),
                    **_tensor_stats(profit_pre),
                },
                "a": {
                    "shape": list(a.shape),
                    **_tensor_stats(a),
                },
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "selected_as_best": val_profit >= best_val_profit,
                "bad_rounds": bad_rounds,
            }
            outer_log.append(log_row)

            w = beta * w0 + (1.0 - beta) * (w * a)
            print(
                f"outer_k {outer_round} done: train_profit={train_profit:.6f}, "
                f"val_profit={val_profit:.6f}, best_val={best_val_profit:.6f}, "
                f"tau={float(tau.detach().cpu().item()):.6f}, "
                f"alpha={float(alpha.detach().cpu().item()):.6f}, "
                f"bad_rounds={bad_rounds}/{outer_patience}, "
                f"W[min/mean/max/std]={log_row['W']['min']:.6f}/"
                f"{log_row['W']['mean']:.6f}/{log_row['W']['max']:.6f}/"
                f"{log_row['W']['std']:.6f}",
                flush=True,
            )

        if bad_rounds >= outer_patience:
            print(f"early stop at outer_k {outer_round}: bad_rounds={bad_rounds}/{outer_patience}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate_split(
        model, bundle.x_train_list, bundle.y_train_reg, bundle.y_train_cls,
        fee=fee, k_max=k_max, k_min=k_min, trading_freq=trading_freq,
        decision_rule=decision_rule, eta_k=eta_k, eta_omega=eta_omega,
    )
    final_val = evaluate_split(
        model, bundle.x_val_list, bundle.y_val_reg, bundle.y_val_cls,
        fee=fee, k_max=k_max, k_min=k_min, trading_freq=trading_freq,
        decision_rule=decision_rule, eta_k=eta_k, eta_omega=eta_omega,
    )
    final_test = evaluate_split(
        model, bundle.x_test_list, bundle.y_test_reg, bundle.y_test_cls,
        fee=fee, k_max=k_max, k_min=k_min, trading_freq=trading_freq,
        decision_rule=decision_rule, eta_k=eta_k, eta_omega=eta_omega,
    )

    model.eval()
    with torch.no_grad():
        test_outputs = model(bundle.x_test_list)
        test_weights, test_selected_k, test_decision_diagnostics = topk_softmax_weights(
            test_outputs["reg_out"],
            k_max=k_max,
            k_min=k_min,
            decision_rule=decision_rule,
            eta_k=eta_k,
            eta_omega=eta_omega,
            return_diagnostics=True,
        )
        test_wealth, _, _ = simulate_return_ratio_wealth(test_weights, bundle.y_test_reg, fee=fee)

    torch.save(model.state_dict(), out_path / "model_seed001.pt")
    np.save(out_path / "W0.npy", w0.detach().cpu().numpy())
    np.save(out_path / "W_final.npy", w.detach().cpu().numpy())
    np.save(out_path / "test_reg_out.npy", test_outputs["reg_out"].detach().cpu().numpy())
    np.save(out_path / "test_cls_out.npy", test_outputs["cls_out"].detach().cpu().numpy())
    np.save(out_path / "test_weights.npy", test_weights.detach().cpu().numpy())
    np.save(out_path / "test_selected_k.npy", test_selected_k.detach().cpu().numpy())
    if isinstance(test_decision_diagnostics.get("hhi"), torch.Tensor):
        np.save(
            out_path / "test_hhi.npy",
            test_decision_diagnostics["hhi"].detach().cpu().numpy(),
        )
    if isinstance(test_decision_diagnostics.get("n_eff"), torch.Tensor):
        np.save(
            out_path / "test_n_eff.npy",
            test_decision_diagnostics["n_eff"].detach().cpu().numpy(),
        )
    np.save(out_path / "test_wealth_curve.npy", test_wealth.detach().cpu().numpy())

    report: Dict[str, object] = {
        "config": {
            "seed": seed,
            "max_outer_rounds": max_outer_rounds,
            "outer_rounds_ran": len(outer_log),
            "inner_epochs": inner_epochs,
            "inner_patience": inner_patience,
            "outer_patience": outer_patience,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "beta": beta,
            "ce_weight": ce_weight,
            "mse_weight": mse_weight,
            "loss_mode": loss_mode,
            "rank_weight": rank_weight,
            "use_contrastive": use_contrastive,
            "contrastive_weight": contrastive_weight,
            "contrastive_margin": contrastive_margin,
            "use_sharpe": use_sharpe,
            "sharpe_weight": sharpe_weight,
            "use_ema_loss_norm": bool(use_ema_loss_norm),
            "ema_alpha": float(ema_alpha),
            "fee": fee,
            "k_max": k_max,
            "k_min": k_min,
            "decision_rule": decision_rule,
            "eta_k": float(eta_k),
            "eta_omega": float(eta_omega),
            "trading_freq": trading_freq,
            "frontend_type": frontend_type,
            "lista_unroll_steps": int(lista_unroll_steps),
            "lbp_unroll_steps": int(lbp_unroll_steps),
            "dropout_rate": float(dropout_rate),
                "use_signature": bool(use_signature),
                "signature_level": int(signature_level) if use_signature else None,
                "weight_update_mode": weight_update_mode,
                "profit_source": profit_source,
                "log_every_epoch": int(log_every_epoch),
                "device": device,
                "profitboost": True,
                "online": False,
                "weight_update": weight_update_desc,
            },
        "meta": bundle.meta,
        "metrics": {
            "train": final_train,
            "val": final_val,
            "test": final_test,
        },
        "outer_log": outer_log,
        "outputs": {
            "model": str(out_path / "model_seed001.pt"),
            "W0": str(out_path / "W0.npy"),
            "W_final": str(out_path / "W_final.npy"),
            "test_reg_out": str(out_path / "test_reg_out.npy"),
            "test_cls_out": str(out_path / "test_cls_out.npy"),
            "test_weights": str(out_path / "test_weights.npy"),
            "test_selected_k": str(out_path / "test_selected_k.npy"),
            "test_hhi": str(out_path / "test_hhi.npy")
            if (out_path / "test_hhi.npy").is_file()
            else None,
            "test_n_eff": str(out_path / "test_n_eff.npy")
            if (out_path / "test_n_eff.npy").is_file()
            else None,
            "test_wealth_curve": str(out_path / "test_wealth_curve.npy"),
            "code_snapshot": str(out_path / "code_snapshot"),
        },
        "code_snapshot_files": len(snapshot_manifest.get("copied_files", [])),
    }
    (out_path / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ProfitBoost smoke test: CE+MSE, no online.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="outputs/torch_proposed/profitboost_ce_mse_seed001")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_outer_rounds", type=int, default=20)
    parser.add_argument("--inner_epochs", type=int, default=5)
    parser.add_argument("--inner_patience", type=int, default=0)
    parser.add_argument("--outer_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument(
        "--loss_mode",
        default="ce_mse",
        choices=[
            "ce_only",
            "ce_mse",
            "ce_rank",
            "ranking_mse",
            "ce_mse_rank",
            "ranking_mse_wrapper",
            "ce_product_rank",
            "product_ranking_mse",
            "ce_mse_product_rank",
        ],
    )
    parser.add_argument("--rank_weight", type=float, default=0.5)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_margin", type=float, default=1.0)
    parser.add_argument("--use_sharpe", action="store_true")
    parser.add_argument("--sharpe_weight", type=float, default=1.0)
    parser.add_argument("--use_ema_loss_norm", action="store_true")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument(
        "--decision_rule",
        default="spread",
        choices=["spread", "hhi_effective", "fixed"],
    )
    parser.add_argument("--eta_k", type=float, default=1.0)
    parser.add_argument("--eta_omega", type=float, default=1.0)
    parser.add_argument("--trading_freq", type=float, default=15.0)
    parser.add_argument(
        "--frontend_type",
        default="dense",
        choices=["dense", "lista", "ML_LISTA_NET", "ista", "ML_ISTA_NET", "fista", "ML_FISTA_NET", "lbp", "LBP_NET"],
    )
    parser.add_argument("--lista_unroll_steps", type=int, default=4)
    parser.add_argument("--lbp_unroll_steps", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--use_signature", action="store_true")
    parser.add_argument("--signature_level", type=int, default=2)
    parser.add_argument(
        "--weight_update_mode",
        default="sample_specific",
        choices=["sample_specific", "scalar_tf_style"],
        help="Offline ProfitBoost weight update. sample_specific is the cleaned version; scalar_tf_style mirrors the legacy TF scalar update.",
    )
    parser.add_argument(
        "--profit_source",
        default="one_step_sum",
        choices=["one_step_sum", "wealth_path"],
        help="Profit source for offline alpha/tau. wealth_path mirrors legacy TF PortfolioSim final-wealth profit.",
    )
    parser.add_argument("--log_every_epoch", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = train_profitboost(
        root=args.root,
        out_dir=args.out_dir,
        seed=args.seed,
        max_outer_rounds=args.max_outer_rounds,
        inner_epochs=args.inner_epochs,
        inner_patience=args.inner_patience,
        outer_patience=args.outer_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
        loss_mode=args.loss_mode,
        rank_weight=args.rank_weight,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        use_sharpe=args.use_sharpe,
        sharpe_weight=args.sharpe_weight,
        use_ema_loss_norm=args.use_ema_loss_norm,
        ema_alpha=args.ema_alpha,
        fee=args.fee,
        k_max=args.k_max,
        k_min=args.k_min,
        decision_rule=args.decision_rule,
        eta_k=args.eta_k,
        eta_omega=args.eta_omega,
        trading_freq=args.trading_freq,
        frontend_type=args.frontend_type,
        lista_unroll_steps=args.lista_unroll_steps,
        lbp_unroll_steps=args.lbp_unroll_steps,
        dropout_rate=args.dropout_rate,
        use_signature=args.use_signature,
        signature_level=args.signature_level,
        weight_update_mode=args.weight_update_mode,
        profit_source=args.profit_source,
        log_every_epoch=args.log_every_epoch,
        device=args.device,
    )
    compact = {
        "outer_rounds_ran": report["config"]["outer_rounds_ran"],
        "metrics": report["metrics"],
        "last_outer_log": report["outer_log"][-1] if report["outer_log"] else None,
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {Path(args.root) / args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
