from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from torch_data import ProposedDataBundle, load_proposed_djia_data
from torch_metrics import calculate_portfolio_metrics
from torch_model import build_model_from_data
from torch_portfolio import simulate_return_ratio_wealth, topk_softmax_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_indices(n: int, batch_size: int, generator: torch.Generator) -> List[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def _take_batch(xs: List[torch.Tensor], idx: torch.Tensor) -> List[torch.Tensor]:
    return [x[idx] for x in xs]


def classification_accuracy(cls_logits: torch.Tensor, y_cls: torch.Tensor) -> float:
    target = y_cls.argmax(dim=-1)
    pred = cls_logits.argmax(dim=-1)
    return float((pred == target).to(torch.float32).mean().item() * 100.0)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    ce_loss: nn.Module,
    mse_loss: nn.Module,
    ce_weight: float,
    mse_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls_logits = outputs["cls_logits"]
    cls_target = y_cls.argmax(dim=-1).to(torch.long)
    ce = ce_loss(cls_logits.reshape(-1, 2), cls_target.reshape(-1))
    mse = mse_loss(outputs["reg_out"], y_reg)
    total = ce_weight * ce + mse_weight * mse
    return total, {
        "loss": float(total.detach().cpu().item()),
        "ce": float(ce.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
    }


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    x_list: List[torch.Tensor],
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    ce_loss: nn.Module,
    mse_loss: nn.Module,
    ce_weight: float,
    mse_weight: float,
    fee: float,
    k_max: int,
    k_min: int,
    trading_freq: float,
) -> Dict[str, object]:
    model.eval()
    outputs = model(x_list)
    _, losses = compute_loss(outputs, y_reg, y_cls, ce_loss, mse_loss, ce_weight, mse_weight)
    acc = classification_accuracy(outputs["cls_logits"], y_cls)
    weights, selected_k = topk_softmax_weights(outputs["reg_out"], k_max=k_max, k_min=k_min, adaptive=True)
    wealth, step_returns, turnover = simulate_return_ratio_wealth(weights, y_reg, fee=fee)
    fin_metrics = calculate_portfolio_metrics(wealth, trading_freq=trading_freq)
    return {
        **losses,
        "acc": acc,
        "final_wealth": float(wealth[-1].detach().cpu().item()),
        "profit_on_1usd": float((wealth[-1] - wealth[0]).detach().cpu().item()),
        "mean_step_return": float(step_returns.detach().cpu().mean().item()),
        "total_turnover": float(turnover.detach().cpu().sum().item()),
        "mean_selected_k": float(selected_k.detach().cpu().to(torch.float32).mean().item()),
        "financial_metrics": fin_metrics,
    }


def train_smoke(
    root: str | Path = ".",
    out_dir: str | Path = "outputs/torch_proposed/smoke_ce_mse_seed001",
    seed: int = 1,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 10,
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
    fee: float = 0.001,
    k_max: int = 5,
    k_min: int = 1,
    trading_freq: float = 15.0,
    frontend_type: str = "dense",
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

    bundle: ProposedDataBundle = load_proposed_djia_data(root=root_path, device=device)
    model = build_model_from_data(bundle, frontend_type=frontend_type).to(device)

    # Build lazy heads before optimizer registration.
    model(_take_batch(bundle.x_train_list, torch.arange(min(batch_size, bundle.y_train_reg.shape[0]), device=device)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    n_train = int(bundle.y_train_reg.shape[0])
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for idx_cpu in _batch_indices(n_train, batch_size, generator):
            idx = idx_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(_take_batch(bundle.x_train_list, idx))
            loss, losses = compute_loss(
                outputs,
                bundle.y_train_reg[idx],
                bundle.y_train_cls[idx],
                ce_loss,
                mse_loss,
                ce_weight,
                mse_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(losses)

        train_loss = float(np.mean([row["loss"] for row in epoch_losses]))
        val_metrics = evaluate_split(
            model,
            bundle.x_val_list,
            bundle.y_val_reg,
            bundle.y_val_cls,
            ce_loss,
            mse_loss,
            ce_weight,
            mse_weight,
            fee,
            k_max,
            k_min,
            trading_freq,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_split(
        model,
        bundle.x_train_list,
        bundle.y_train_reg,
        bundle.y_train_cls,
        ce_loss,
        mse_loss,
        ce_weight,
        mse_weight,
        fee,
        k_max,
        k_min,
        trading_freq,
    )
    val_metrics = evaluate_split(
        model,
        bundle.x_val_list,
        bundle.y_val_reg,
        bundle.y_val_cls,
        ce_loss,
        mse_loss,
        ce_weight,
        mse_weight,
        fee,
        k_max,
        k_min,
        trading_freq,
    )
    test_metrics = evaluate_split(
        model,
        bundle.x_test_list,
        bundle.y_test_reg,
        bundle.y_test_cls,
        ce_loss,
        mse_loss,
        ce_weight,
        mse_weight,
        fee,
        k_max,
        k_min,
        trading_freq,
    )

    model.eval()
    with torch.no_grad():
        test_outputs = model(bundle.x_test_list)
        test_weights, _ = topk_softmax_weights(test_outputs["reg_out"], k_max=k_max, k_min=k_min, adaptive=True)
        test_wealth, _, _ = simulate_return_ratio_wealth(test_weights, bundle.y_test_reg, fee=fee)

    torch.save(model.state_dict(), out_path / "model_seed001.pt")
    np.save(out_path / "test_reg_out.npy", test_outputs["reg_out"].detach().cpu().numpy())
    np.save(out_path / "test_cls_out.npy", test_outputs["cls_out"].detach().cpu().numpy())
    np.save(out_path / "test_weights.npy", test_weights.detach().cpu().numpy())
    np.save(out_path / "test_wealth_curve.npy", test_wealth.detach().cpu().numpy())

    report: Dict[str, object] = {
        "config": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_ran": len(history),
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "ce_weight": ce_weight,
            "mse_weight": mse_weight,
            "fee": fee,
            "k_max": k_max,
            "k_min": k_min,
            "trading_freq": trading_freq,
            "frontend_type": frontend_type,
            "device": device,
            "profitboost": False,
            "online": False,
        },
        "meta": bundle.meta,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
        "outputs": {
            "model": str(out_path / "model_seed001.pt"),
            "test_reg_out": str(out_path / "test_reg_out.npy"),
            "test_cls_out": str(out_path / "test_cls_out.npy"),
            "test_weights": str(out_path / "test_weights.npy"),
            "test_wealth_curve": str(out_path / "test_wealth_curve.npy"),
        },
    }
    (out_path / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: CE+MSE, no ProfitBoost, no online.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="outputs/torch_proposed/smoke_ce_mse_seed001")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--trading_freq", type=float, default=15.0)
    parser.add_argument(
        "--frontend_type",
        default="dense",
        choices=["dense", "lista", "ML_LISTA_NET", "ista", "ML_ISTA_NET", "fista", "ML_FISTA_NET", "lbp", "LBP_NET"],
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = train_smoke(
        root=args.root,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
        fee=args.fee,
        k_max=args.k_max,
        k_min=args.k_min,
        trading_freq=args.trading_freq,
        frontend_type=args.frontend_type,
        device=args.device,
    )
    compact = {
        "epochs_ran": report["config"]["epochs_ran"],
        "train": report["metrics"]["train"],
        "val": report["metrics"]["val"],
        "test": report["metrics"]["test"],
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {Path(args.root) / args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
