from __future__ import annotations

import torch

from torch_losses import EMALossNormalizer
from torch_model_paper_sweep_frontend_solver_20260619 import (
    FRONTEND_SPECS,
    OHLCGRUPaperSweepModel,
    sparse_fan_in_std,
)
from torch_portfolio import adaptive_k_hhi_effective, topk_softmax_weights
from torch_train_profitboost import rescale_profit


def test_formal_model_tensor_shapes() -> None:
    model = OHLCGRUPaperSweepModel(
        sector_sizes=[7, 5, 4, 6, 8],
        n_assets=30,
        output_len=1,
        seq_len=15,
        features_per_asset=4,
        spec=FRONTEND_SPECS["F3_a2_5day"],
        solver_type="lista",
        hidden_dim=64,
        num_heads=4,
        n_attention_layers=3,
        unroll_steps=6,
        dropout_rate=0.2,
    ).eval()
    inputs = [torch.zeros(2, 15, size * 4) for size in [7, 5, 4, 6, 8]]
    with torch.no_grad():
        outputs = model(inputs)
    assert tuple(outputs["reg_out"].shape) == (2, 1, 30)
    assert tuple(outputs["cls_logits"].shape) == (2, 1, 30, 2)
    assert tuple(outputs["con_out"].shape) == (2, 15, 30 * 32)


def test_sparse_initialization_and_hhi_bounds() -> None:
    assert sparse_fan_in_std(1, (4, 4)) == 0.025
    scores = torch.tensor([[0.9, 0.4, 0.2, 0.1, -0.1]])
    selected_k, diagnostics = adaptive_k_hhi_effective(
        scores,
        k_min=1,
        k_max=5,
        eta_k=0.5,
        return_diagnostics=True,
    )
    assert int(selected_k.min()) >= 1
    assert int(selected_k.max()) <= 5
    assert torch.all(diagnostics["hhi"] > 0)


def test_final_weights_and_training_utilities() -> None:
    scores = torch.tensor([[[0.9, 0.4, 0.2, 0.1, -0.1]]])
    weights, selected_k = topk_softmax_weights(
        scores,
        k_min=1,
        k_max=5,
        decision_rule="hhi_effective",
        eta_k=0.5,
        eta_omega=2.0,
    )
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    assert torch.all(weights >= 0)
    assert selected_k.shape == (1,)

    normalizer = EMALossNormalizer(alpha=0.1)
    normalizer.denominator("ce", torch.tensor(2.0))
    before = normalizer.snapshot()
    with normalizer.frozen():
        normalizer.denominator("ce", torch.tensor(4.0))
    assert normalizer.snapshot() == before

    tau = rescale_profit(torch.tensor([1.0]), torch.tensor([2.0]))
    torch.testing.assert_close(tau, torch.tensor([0.25]))
