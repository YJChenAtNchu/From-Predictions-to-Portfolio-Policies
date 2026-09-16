from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from experiment_snapshot import save_code_snapshot
from torch_data_formal_split import FormalProposedDataBundle, load_proposed_djia_formal_data
from torch_model import build_model_from_data
from torch_portfolio import simulate_return_ratio_wealth, topk_softmax_weights
from torch_train_profitboost import (
    _tensor_stats,
    evaluate_split,
    get_w0_from_rate,
    one_step_portfolio_profit,
    portfolio_path_profit,
    rescale_profit,
    train_one_outer_round,
)
from torch_train_smoke import _take_batch, set_seed


def train_profitboost_formal_val(
    data_root: str | Path,
    out_dir: str | Path,
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
    trading_freq: float = 5.0,
    frontend_type: str = "lista",
    lista_unroll_steps: int = 4,
    lbp_unroll_steps: int = 4,
    dropout_rate: float = 0.0,
    use_signature: bool = False,
    signature_level: int = 2,
    weight_update_mode: str = "sample_specific",
    profit_source: str = "one_step_sum",
    inner_train_val_split: float = 0.9,
    log_every_epoch: int = 0,
    device: str = "auto",
) -> Dict[str, object]:
    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = Path(__file__).resolve().parent
    data_path = Path(data_root)
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    snapshot_manifest = save_code_snapshot(
        out_path,
        root=project_root,
        extra_files=[
            Path(__file__),
            project_root / "torch_data_formal_split.py",
            project_root / "build_djia30_2026_seq15_tar5_stride5_ratio_formal_split_dataset.py",
        ],
        note="formal outer-validation ProfitBoost training; final test is not evaluated",
    )

    bundle: FormalProposedDataBundle = load_proposed_djia_formal_data(
        root=data_path,
        inner_train_val_split=inner_train_val_split,
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

    # Build lazy heads before optimizer registration.
    warmup_idx = torch.arange(min(batch_size, bundle.y_train_reg.shape[0]), device=device)
    model(_take_batch(bundle.x_train_list, warmup_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    w0 = get_w0_from_rate(bundle.y_train_rate).to(device)
    w = w0.clone()

    if profit_source == "one_step_sum":
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

    best_inner_val_profit = -float("inf")
    best_state = None
    bad_rounds = 0
    outer_log = []
    weight_update_desc = ""

    for outer_round in range(max_outer_rounds):
        print(f"formal outer_k {outer_round} start", flush=True)
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
                weight_update_desc = "sample-specific a=exp(-max(alpha,0)*tanh(sample_profit_pre))"
            elif weight_update_mode == "scalar_tf_style":
                a = torch.exp(-alpha_weight * torch.tanh(train_profit))
                weight_update_desc = f"scalar TF-style a=exp(-max(alpha,0)*tanh(total_train_profit)), profit_source={profit_source}"
            else:
                raise ValueError(f"Unsupported weight_update_mode={weight_update_mode}")

            inner_train_metrics = evaluate_split(
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
            inner_val_metrics = evaluate_split(
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
            outer_val_metrics = evaluate_split(
                model,
                bundle.x_outer_val_list,
                bundle.y_outer_val_reg,
                bundle.y_outer_val_cls,
                fee=fee,
                k_max=k_max,
                k_min=k_min,
                trading_freq=trading_freq,
                decision_rule=decision_rule,
                eta_k=eta_k,
                eta_omega=eta_omega,
            )

            inner_val_profit = float(inner_val_metrics["profit_on_1usd"])
            selected_as_best = False
            if inner_val_profit > best_inner_val_profit:
                best_inner_val_profit = inner_val_profit
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_rounds = 0
                selected_as_best = True
            else:
                bad_rounds += 1

            log_row = {
                "outer_round": outer_round,
                "fit_loss": fit_log,
                "train_profit": float(train_profit.detach().cpu().item()),
                "inner_val_profit": inner_val_profit,
                "outer_val_profit": float(outer_val_metrics["profit_on_1usd"]),
                "tau": float(tau.detach().cpu().item()),
                "alpha": float(alpha.detach().cpu().item()),
                "alpha_weight": float(alpha_weight.detach().cpu().item()),
                "r_max": float(r_max.detach().cpu().item()),
                "W": _tensor_stats(w),
                "W_cur": _tensor_stats(w_cur),
                "profit_pre": {"shape": list(profit_pre.shape), **_tensor_stats(profit_pre)},
                "a": {"shape": list(a.shape), **_tensor_stats(a)},
                "inner_train_metrics": inner_train_metrics,
                "inner_validation_metrics": inner_val_metrics,
                "outer_validation_metrics": outer_val_metrics,
                "selected_as_best": selected_as_best,
                "bad_rounds": bad_rounds,
            }
            outer_log.append(log_row)

            w = beta * w0 + (1.0 - beta) * (w * a)
            print(
                f"formal outer_k {outer_round} done: train_profit={train_profit:.6f}, "
                f"inner_val_profit={inner_val_profit:.6f}, "
                f"outer_val_profit={outer_val_metrics['profit_on_1usd']:.6f}, "
                f"best_inner_val={best_inner_val_profit:.6f}, "
                f"tau={float(tau.detach().cpu().item()):.6f}, "
                f"alpha={float(alpha.detach().cpu().item()):.6f}, "
                f"bad_rounds={bad_rounds}/{outer_patience}",
                flush=True,
            )

        if bad_rounds >= outer_patience:
            print(f"formal early stop at outer_k {outer_round}: bad_rounds={bad_rounds}/{outer_patience}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_inner_train = evaluate_split(
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
        return_outputs=True,
    )
    final_inner_val = evaluate_split(
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
        return_outputs=True,
    )
    final_outer_val = evaluate_split(
        model,
        bundle.x_outer_val_list,
        bundle.y_outer_val_reg,
        bundle.y_outer_val_cls,
        fee=fee,
        k_max=k_max,
        k_min=k_min,
        trading_freq=trading_freq,
        decision_rule=decision_rule,
        eta_k=eta_k,
        eta_omega=eta_omega,
        return_outputs=True,
    )

    def extract_saved_tensors(metrics: Dict[str, object]) -> Dict[str, object]:
        return {
            "outputs": metrics.pop("_outputs"),
            "weights": metrics.pop("_weights"),
            "selected_k": metrics.pop("_selected_k"),
            "decision_diagnostics": metrics.pop("_decision_diagnostics"),
            "wealth": metrics.pop("_wealth"),
        }

    final_tensors = {
        "inner_train": extract_saved_tensors(final_inner_train),
        "inner_validation": extract_saved_tensors(final_inner_val),
        "outer_validation": extract_saved_tensors(final_outer_val),
    }

    model_path = out_path / f"model_seed{seed:03d}.pt"
    torch.save(model.state_dict(), model_path)
    np.save(out_path / "W0.npy", w0.detach().cpu().numpy())
    np.save(out_path / "W_final.npy", w.detach().cpu().numpy())
    saved_split_outputs: Dict[str, Dict[str, str | None]] = {}
    for split_name, tensors in final_tensors.items():
        file_prefix = {
            "inner_train": "train",
            "inner_validation": "validation",
            "outer_validation": "outer_val",
        }[split_name]
        outputs = tensors["outputs"]
        diagnostics = tensors["decision_diagnostics"]
        paths = {
            "reg_out": out_path / f"{file_prefix}_reg_out.npy",
            "cls_out": out_path / f"{file_prefix}_cls_out.npy",
            "weights": out_path / f"{file_prefix}_weights.npy",
            "selected_k": out_path / f"{file_prefix}_selected_k.npy",
            "wealth_curve": out_path / f"{file_prefix}_wealth_curve.npy",
            "hhi": out_path / f"{file_prefix}_hhi.npy",
            "n_eff": out_path / f"{file_prefix}_n_eff.npy",
        }
        np.save(paths["reg_out"], outputs["reg_out"].detach().cpu().numpy())
        np.save(paths["cls_out"], outputs["cls_out"].detach().cpu().numpy())
        np.save(paths["weights"], tensors["weights"].detach().cpu().numpy())
        np.save(paths["selected_k"], tensors["selected_k"].detach().cpu().numpy())
        np.save(paths["wealth_curve"], tensors["wealth"].detach().cpu().numpy())
        hhi = diagnostics.get("hhi")
        n_eff = diagnostics.get("n_eff")
        if isinstance(hhi, torch.Tensor):
            np.save(paths["hhi"], hhi.detach().cpu().numpy())
        if isinstance(n_eff, torch.Tensor):
            np.save(paths["n_eff"], n_eff.detach().cpu().numpy())
        saved_split_outputs[split_name] = {
            key: str(path) if path.is_file() else None for key, path in paths.items()
        }

    report: Dict[str, object] = {
        "config": {
            "seed": int(seed),
            "stage": "outer_validation_search",
            "final_test_evaluated": False,
            "max_outer_rounds": int(max_outer_rounds),
            "outer_rounds_ran": len(outer_log),
            "inner_epochs": int(inner_epochs),
            "inner_patience": int(inner_patience),
            "outer_patience": int(outer_patience),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "beta": float(beta),
            "ce_weight": float(ce_weight),
            "mse_weight": float(mse_weight),
            "loss_mode": loss_mode,
            "rank_weight": float(rank_weight),
            "use_contrastive": bool(use_contrastive),
            "contrastive_weight": float(contrastive_weight),
            "contrastive_margin": float(contrastive_margin),
            "use_sharpe": bool(use_sharpe),
            "sharpe_weight": float(sharpe_weight),
            "use_ema_loss_norm": bool(use_ema_loss_norm),
            "ema_alpha": float(ema_alpha),
            "fee": float(fee),
            "k_max": int(k_max),
            "k_min": int(k_min),
            "decision_rule": decision_rule,
            "eta_k": float(eta_k),
            "eta_omega": float(eta_omega),
            "trading_freq": float(trading_freq),
            "frontend_type": frontend_type,
            "lista_unroll_steps": int(lista_unroll_steps),
            "lbp_unroll_steps": int(lbp_unroll_steps),
            "dropout_rate": float(dropout_rate),
            "use_signature": bool(use_signature),
            "signature_level": int(signature_level) if use_signature else None,
            "weight_update_mode": weight_update_mode,
            "profit_source": profit_source,
            "weight_update": weight_update_desc,
            "inner_train_val_split": float(inner_train_val_split),
            "log_every_epoch": int(log_every_epoch),
            "device": device,
            "profitboost": True,
            "online": False,
        },
        "data_protocol": {
            "outer_train": "2009-2020; split by target_end_date",
            "outer_validation": "2021-2022; used for configuration selection",
            "final_test": "2023-2025; intentionally not evaluated in this script",
            "split_by": "target_end_date",
        },
        "meta": bundle.meta,
        "inner_validation": final_inner_val,
        "outer_validation": final_outer_val,
        "metrics": {
            "inner_train": final_inner_train,
            "inner_validation": final_inner_val,
            "outer_validation": final_outer_val,
            "final_test": None,
        },
        "outer_log": outer_log,
        "outputs": {
            "model": str(model_path),
            "W0": str(out_path / "W0.npy"),
            "W_final": str(out_path / "W_final.npy"),
            "splits": saved_split_outputs,
            "code_snapshot": str(out_path / "code_snapshot"),
        },
        "code_snapshot_files": len(snapshot_manifest.get("copied_files", [])),
    }
    (out_path / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal split ProfitBoost outer-validation training.")
    parser.add_argument(
        "--data_root",
        "--root",
        dest="data_root",
        default="data/djia30_2026_seq15_tar5_stride5_ratio_split2020_2022_2025",
    )
    parser.add_argument("--out_dir", default="outputs/torch_proposed/formal_val_smoke_seed001")
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
    parser.add_argument("--trading_freq", type=float, default=5.0)
    parser.add_argument(
        "--frontend_type",
        default="lista",
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
    )
    parser.add_argument(
        "--profit_source",
        default="one_step_sum",
        choices=["one_step_sum", "wealth_path"],
    )
    parser.add_argument("--inner_train_val_split", type=float, default=0.9)
    parser.add_argument("--log_every_epoch", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = train_profitboost_formal_val(
        data_root=args.data_root,
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
        inner_train_val_split=args.inner_train_val_split,
        log_every_epoch=args.log_every_epoch,
        device=args.device,
    )
    compact = {
        "outer_rounds_ran": report["config"]["outer_rounds_ran"],
        "inner_validation": report["inner_validation"],
        "outer_validation": report["outer_validation"],
        "final_test": report["metrics"]["final_test"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
