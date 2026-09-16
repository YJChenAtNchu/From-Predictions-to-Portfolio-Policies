from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from torch_model import AttentionBlock, SequenceBatchNorm
from torch_model_asset_aligned_20260614 import _conv_same, _deconv_to_shape


@dataclass(frozen=True)
class SparseKernelSpec:
    first_kernel: tuple[int, int]
    first_stride: tuple[int, int]
    layer2_kernel: tuple[int, int]
    layer2_stride: tuple[int, int]
    layer3_kernel: tuple[int, int]
    layer3_stride: tuple[int, int]


SPARSE_INIT = "fan_in_gaussian"
SPARSE_INIT_SCALE = 0.1
SPARSE_INIT_FAN_IN = "in_channels * kernel_height * kernel_width"


def sparse_fan_in_std(in_channels: int, kernel: tuple[int, int], init_scale: float = SPARSE_INIT_SCALE) -> float:
    kh, kw = int(kernel[0]), int(kernel[1])
    fan_in = max(1, int(in_channels) * kh * kw)
    return float(init_scale) / math.sqrt(float(fan_in))


def init_sparse_operator_(
    param: torch.Tensor,
    *,
    in_channels: int,
    kernel: tuple[int, int],
    init_scale: float = SPARSE_INIT_SCALE,
) -> None:
    nn.init.normal_(param, mean=0.0, std=sparse_fan_in_std(in_channels, kernel, init_scale))


class AssetAlignedSparseSolver(nn.Module):
    """Asset-aligned sparse frontend with switchable ISTA/LISTA/FISTA/LBP solvers."""

    def __init__(
        self,
        *,
        seq_len: int,
        features_per_asset: int,
        spec: SparseKernelSpec,
        solver_type: str,
        channels: int = 32,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        if solver_type not in {"ista", "lista", "fista", "lbp"}:
            raise ValueError(f"Unsupported solver_type={solver_type}")
        if spec.first_kernel[1] != features_per_asset:
            raise ValueError(
                "First-layer kernel width must match features_per_asset to preserve asset boundaries: "
                f"kernel={spec.first_kernel}, F={features_per_asset}"
            )
        self.seq_len = int(seq_len)
        self.features_per_asset = int(features_per_asset)
        self.spec = spec
        self.solver_type = solver_type
        self.unroll_steps = int(unroll_steps)
        self.channels = int(channels)

        if solver_type == "lista":
            self.B1 = nn.Parameter(torch.empty(channels, 1, *spec.first_kernel))
            self.B2 = nn.Parameter(torch.empty(channels, channels, *spec.layer2_kernel))
            self.B3 = nn.Parameter(torch.empty(channels, channels, *spec.layer3_kernel))
            self.W1 = nn.Parameter(torch.empty(channels, 1, *spec.first_kernel))
            self.W2 = nn.Parameter(torch.empty(channels, channels, *spec.layer2_kernel))
            self.W3 = nn.Parameter(torch.empty(channels, channels, *spec.layer3_kernel))
            self._reset_pair("B1", "W1", in_channels=1, kernel=spec.first_kernel)
            self._reset_pair("B2", "W2", in_channels=channels, kernel=spec.layer2_kernel)
            self._reset_pair("B3", "W3", in_channels=channels, kernel=spec.layer3_kernel)
        else:
            self.W1 = nn.Parameter(torch.empty(channels, 1, *spec.first_kernel))
            self.W2 = nn.Parameter(torch.empty(channels, channels, *spec.layer2_kernel))
            self.W3 = nn.Parameter(torch.empty(channels, channels, *spec.layer3_kernel))
            init_sparse_operator_(self.W1, in_channels=1, kernel=spec.first_kernel)
            init_sparse_operator_(self.W2, in_channels=channels, kernel=spec.layer2_kernel)
            init_sparse_operator_(self.W3, in_channels=channels, kernel=spec.layer3_kernel)
            self.c1 = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.c2 = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.c3 = nn.Parameter(torch.ones(1, 1, 1, 1))

        self.bias1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _reset_pair(
        self,
        b_name: str,
        w_name: str,
        *,
        in_channels: int,
        kernel: tuple[int, int],
    ) -> None:
        init_sparse_operator_(getattr(self, b_name), in_channels=in_channels, kernel=kernel)
        init_sparse_operator_(getattr(self, w_name), in_channels=in_channels, kernel=kernel)

    def _initial_lista(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g1 = torch.relu(_conv_same(x_img, self.B1, stride=self.spec.first_stride) + self.bias1)
        g2 = torch.relu(_conv_same(g1, self.B2, stride=self.spec.layer2_stride) + self.bias2)
        g3 = torch.relu(_conv_same(g2, self.B3, stride=self.spec.layer3_stride) + self.bias3)
        return g1, g2, g3

    def _initial_tied(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g1 = torch.relu(self.c1 * _conv_same(x_img, self.W1, stride=self.spec.first_stride) + self.bias1)
        g2 = torch.relu(self.c2 * _conv_same(g1, self.W2, stride=self.spec.layer2_stride) + self.bias2)
        g3 = torch.relu(self.c3 * _conv_same(g2, self.W3, stride=self.spec.layer3_stride) + self.bias3)
        return g1, g2, g3

    def _forward_lista(self, x_img: torch.Tensor) -> torch.Tensor:
        g1, g2, g3 = self._initial_lista(x_img)
        for _ in range(self.unroll_steps):
            g2_back = _deconv_to_shape(g3, self.B3, g2, stride=self.spec.layer3_stride)
            g1_back = _deconv_to_shape(g2_back, self.B2, g1, stride=self.spec.layer2_stride)
            x_rec = _deconv_to_shape(g1_back, self.W1, x_img, stride=self.spec.first_stride)
            g1 = torch.relu(
                g1_back
                - _conv_same(x_rec, self.W1, stride=self.spec.first_stride)
                + _conv_same(x_img, self.B1, stride=self.spec.first_stride)
                + self.bias1
            )
            g2_rec = _deconv_to_shape(g2_back, self.W2, g1, stride=self.spec.layer2_stride)
            g2 = torch.relu(
                g2_back
                - _conv_same(g2_rec, self.W2, stride=self.spec.layer2_stride)
                + _conv_same(g1, self.B2, stride=self.spec.layer2_stride)
                + self.bias2
            )
            g3_rec = _deconv_to_shape(g3, self.W3, g2, stride=self.spec.layer3_stride)
            g3 = torch.relu(
                g3
                - _conv_same(g3_rec, self.W3, stride=self.spec.layer3_stride)
                + _conv_same(g2, self.B3, stride=self.spec.layer3_stride)
                + self.bias3
            )
        return g3

    def _forward_ista(self, x_img: torch.Tensor) -> torch.Tensor:
        g1, g2, g3 = self._initial_tied(x_img)
        for _ in range(self.unroll_steps):
            g2_back = _deconv_to_shape(g3, self.W3, g2, stride=self.spec.layer3_stride)
            g1_back = _deconv_to_shape(g2_back, self.W2, g1, stride=self.spec.layer2_stride)
            g1_res = _deconv_to_shape(g1_back, self.W1, x_img, stride=self.spec.first_stride) - x_img
            g1 = torch.relu(g1_back - self.c1 * _conv_same(g1_res, self.W1, stride=self.spec.first_stride) + self.bias1)

            g2_res = _deconv_to_shape(g2_back, self.W2, g1, stride=self.spec.layer2_stride) - g1
            g2 = torch.relu(g2_back - self.c2 * _conv_same(g2_res, self.W2, stride=self.spec.layer2_stride) + self.bias2)

            g3_res = _deconv_to_shape(g3, self.W3, g2, stride=self.spec.layer3_stride) - g2
            g3 = torch.relu(g3 - self.c3 * _conv_same(g3_res, self.W3, stride=self.spec.layer3_stride) + self.bias3)
        return g3

    def _forward_fista(self, x_img: torch.Tensor) -> torch.Tensor:
        g1, g2, g3 = self._initial_tied(x_img)
        g3_prev = g3
        t = 1.0
        for _ in range(self.unroll_steps):
            t_prev = t
            t = (1.0 + float(np.sqrt(1.0 + 4.0 * t_prev**2))) / 2.0
            z = g3 + ((t_prev - 1.0) / t) * (g3 - g3_prev)
            g3_prev = g3
            g2_back = _deconv_to_shape(z, self.W3, g2, stride=self.spec.layer3_stride)
            g1_back = _deconv_to_shape(g2_back, self.W2, g1, stride=self.spec.layer2_stride)
            g1_res = _deconv_to_shape(g1_back, self.W1, x_img, stride=self.spec.first_stride) - x_img
            g1 = torch.relu(g1_back - self.c1 * _conv_same(g1_res, self.W1, stride=self.spec.first_stride) + self.bias1)
            g2_res = _deconv_to_shape(g2_back, self.W2, g1, stride=self.spec.layer2_stride) - g1
            g2 = torch.relu(g2_back - self.c2 * _conv_same(g2_res, self.W2, stride=self.spec.layer2_stride) + self.bias2)
            g3_res = _deconv_to_shape(z, self.W3, g2, stride=self.spec.layer3_stride) - g2
            g3 = torch.relu(z - self.c3 * _conv_same(g3_res, self.W3, stride=self.spec.layer3_stride) + self.bias3)
        return g3

    def _forward_lbp(self, x_img: torch.Tensor) -> torch.Tensor:
        g1 = torch.relu(self.c1 * _conv_same(x_img, self.W1, stride=self.spec.first_stride) + self.bias1)
        for _ in range(self.unroll_steps):
            residual = _deconv_to_shape(g1, self.W1, x_img, stride=self.spec.first_stride) - x_img
            g1 = torch.relu(g1 - self.c1 * _conv_same(residual, self.W1, stride=self.spec.first_stride) + self.bias1)

        g2 = torch.relu(self.c2 * _conv_same(g1, self.W2, stride=self.spec.layer2_stride) + self.bias2)
        for _ in range(self.unroll_steps):
            residual = _deconv_to_shape(g2, self.W2, g1, stride=self.spec.layer2_stride) - g1
            g2 = torch.relu(g2 - self.c2 * _conv_same(residual, self.W2, stride=self.spec.layer2_stride) + self.bias2)

        g3 = torch.relu(self.c3 * _conv_same(g2, self.W3, stride=self.spec.layer3_stride) + self.bias3)
        for _ in range(self.unroll_steps):
            residual = _deconv_to_shape(g3, self.W3, g2, stride=self.spec.layer3_stride) - g2
            g3 = torch.relu(g3 - self.c3 * _conv_same(residual, self.W3, stride=self.spec.layer3_stride) + self.bias3)
        return g3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,A*F), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got {x.shape[1]}")
        if x.shape[2] % self.features_per_asset:
            raise ValueError(f"Width {x.shape[2]} not divisible by F={self.features_per_asset}")
        expected_assets = x.shape[2] // self.features_per_asset
        x_img = x.unsqueeze(1)
        if self.solver_type == "lista":
            g3 = self._forward_lista(x_img)
        elif self.solver_type == "ista":
            g3 = self._forward_ista(x_img)
        elif self.solver_type == "fista":
            g3 = self._forward_fista(x_img)
        else:
            g3 = self._forward_lbp(x_img)
        if g3.shape[-1] != expected_assets:
            raise RuntimeError(f"Asset alignment failed: expected width {expected_assets}, got {g3.shape[-1]}")
        return g3.permute(0, 2, 3, 1).contiguous()


class OHLCSectorGRUEncoder(nn.Module):
    def __init__(
        self,
        *,
        sector_size: int,
        seq_len: int,
        features_per_asset: int,
        spec: SparseKernelSpec,
        solver_type: str,
        hidden_dim: int = 64,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = AssetAlignedSparseSolver(
            seq_len=seq_len,
            features_per_asset=features_per_asset,
            spec=spec,
            solver_type=solver_type,
            unroll_steps=unroll_steps,
        )
        self.gru = nn.GRU(int(sector_size) * 32, hidden_dim, batch_first=True)
        self.bn = SequenceBatchNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_assets = self.frontend(x)
        sparse_sector = sparse_assets.flatten(start_dim=2)
        encoded, _ = self.gru(sparse_sector)
        encoded = self.dropout(self.act(self.bn(encoded)))
        return encoded, sparse_sector


class OHLCGRUPaperSweepModel(nn.Module):
    def __init__(
        self,
        *,
        sector_sizes: List[int],
        n_assets: int,
        output_len: int,
        seq_len: int,
        features_per_asset: int,
        spec: SparseKernelSpec,
        solver_type: str,
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_attention_layers: int = 3,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)
        self.sector_encoders = nn.ModuleList(
            [
                OHLCSectorGRUEncoder(
                    sector_size=size,
                    seq_len=seq_len,
                    features_per_asset=features_per_asset,
                    spec=spec,
                    solver_type=solver_type,
                    hidden_dim=hidden_dim,
                    unroll_steps=unroll_steps,
                    dropout_rate=dropout_rate,
                )
                for size in sector_sizes
            ]
        )
        concat_dim = hidden_dim * len(sector_sizes)
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(concat_dim, num_heads, dropout_rate=dropout_rate) for _ in range(n_attention_layers)]
        )
        self.head_dropout = nn.Dropout(dropout_rate)
        self.reg_head: nn.Linear | None = None
        self.cls_heads = nn.ModuleList()
        self.flatten_dim: int | None = None

    def _build_heads(self, x: torch.Tensor) -> None:
        flat_dim = int(x.shape[1] * x.shape[2])
        if self.flatten_dim == flat_dim:
            return
        if self.flatten_dim is not None:
            raise ValueError(f"Head input changed from {self.flatten_dim} to {flat_dim}")
        self.flatten_dim = flat_dim
        self.reg_head = nn.Linear(flat_dim, self.output_len * self.n_assets).to(x.device)
        self.cls_heads = nn.ModuleList(
            [nn.Linear(flat_dim, self.output_len * 2).to(x.device) for _ in range(self.n_assets)]
        )

    def forward(self, x_sector_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded, sparse = [], []
        for encoder, x_sector in zip(self.sector_encoders, x_sector_list):
            sector_encoded, sector_sparse = encoder(x_sector)
            encoded.append(sector_encoded)
            sparse.append(sector_sparse)
        x = torch.cat(encoded, dim=-1)
        for block in self.attention_blocks:
            x = block(x)
        self._build_heads(x)
        flat = self.head_dropout(x.flatten(start_dim=1))
        if self.reg_head is None:
            raise RuntimeError("Heads were not initialized")
        reg_out = self.reg_head(flat).reshape(x.shape[0], self.output_len, self.n_assets)
        cls_logits = torch.cat(
            [head(flat).reshape(x.shape[0], self.output_len, 1, 2) for head in self.cls_heads],
            dim=2,
        )
        return {
            "reg_out": reg_out,
            "cls_logits": cls_logits,
            "cls_out": torch.softmax(cls_logits, dim=-1),
            "con_out": torch.cat(sparse, dim=-1),
        }


class SignatureSectorAttentionPaperSweepModel(nn.Module):
    def __init__(
        self,
        *,
        sector_sizes: List[int],
        n_assets: int,
        output_len: int,
        features_per_asset: int,
        spec: SparseKernelSpec,
        solver_type: str,
        embed_dim: int = 32,
        num_heads: int = 4,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.sector_sizes = [int(size) for size in sector_sizes]
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)
        self.embed_dim = int(embed_dim)
        self.frontends = nn.ModuleList(
            [
                AssetAlignedSparseSolver(
                    seq_len=1,
                    features_per_asset=features_per_asset,
                    spec=spec,
                    solver_type=solver_type,
                    channels=embed_dim,
                    unroll_steps=unroll_steps,
                )
                for _ in sector_sizes
            ]
        )
        self.intra_sector_attention = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in sector_sizes]
        )
        self.inter_sector_attention = AttentionBlock(embed_dim, num_heads, dropout_rate=dropout_rate)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.reg_head = nn.Linear(embed_dim, self.output_len)
        self.cls_head = nn.Linear(embed_dim, self.output_len * 2)

    def forward(self, x_sector_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        asset_groups: list[torch.Tensor] = []
        sector_tokens: list[torch.Tensor] = []
        sparse_groups: list[torch.Tensor] = []
        for expected_assets, frontend, attention, x_sector in zip(
            self.sector_sizes,
            self.frontends,
            self.intra_sector_attention,
            x_sector_list,
        ):
            sparse = frontend(x_sector)
            if sparse.shape[1] != 1 or sparse.shape[2] != expected_assets:
                raise RuntimeError(f"Unexpected FS sparse shape: {tuple(sparse.shape)}")
            asset_tokens = attention(sparse[:, 0, :, :])
            asset_groups.append(asset_tokens)
            sector_tokens.append(asset_tokens.mean(dim=1))
            sparse_groups.append(sparse[:, 0, :, :])
        sector_stack = torch.stack(sector_tokens, dim=1)
        sector_context = self.inter_sector_attention(sector_stack)
        fused_groups = []
        for sector_idx, asset_tokens in enumerate(asset_groups):
            context = sector_context[:, sector_idx : sector_idx + 1, :].expand(-1, asset_tokens.shape[1], -1)
            fused_groups.append(self.fusion(torch.cat([asset_tokens, context], dim=-1)))
        fused_assets = torch.cat(fused_groups, dim=1)
        reg_asset_first = self.reg_head(fused_assets)
        reg_out = reg_asset_first.permute(0, 2, 1).contiguous()
        cls_asset_first = self.cls_head(fused_assets).reshape(
            fused_assets.shape[0],
            self.n_assets,
            self.output_len,
            2,
        )
        cls_logits = cls_asset_first.permute(0, 2, 1, 3).contiguous()
        return {
            "reg_out": reg_out,
            "cls_logits": cls_logits,
            "cls_out": torch.softmax(cls_logits, dim=-1),
            "con_out": torch.cat(sparse_groups, dim=1),
            "asset_tokens": torch.cat(asset_groups, dim=1),
            "sector_tokens": sector_stack,
            "sector_context": sector_context,
        }


FRONTEND_SPECS: dict[str, SparseKernelSpec] = {
    "F2_temporal_independent": SparseKernelSpec(
        first_kernel=(1, 4),
        first_stride=(1, 4),
        layer2_kernel=(1, 1),
        layer2_stride=(1, 1),
        layer3_kernel=(1, 1),
        layer3_stride=(1, 1),
    ),
    "F3_a2_5day": SparseKernelSpec(
        first_kernel=(4, 4),
        first_stride=(1, 4),
        layer2_kernel=(2, 1),
        layer2_stride=(1, 1),
        layer3_kernel=(1, 1),
        layer3_stride=(1, 1),
    ),
    "FS_sl_hsa_l2": SparseKernelSpec(
        first_kernel=(1, 20),
        first_stride=(1, 20),
        layer2_kernel=(1, 1),
        layer2_stride=(1, 1),
        layer3_kernel=(1, 1),
        layer3_stride=(1, 1),
    ),
}


def build_paper_sweep_model_from_data(
    bundle,
    *,
    frontend_family: str,
    solver_type: str,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_attention_layers: int = 3,
    lista_unroll_steps: int = 4,
    lbp_unroll_steps: int = 4,
    dropout_rate: float = 0.1,
    **_: object,
) -> nn.Module:
    if frontend_family not in FRONTEND_SPECS:
        raise ValueError(f"Unsupported frontend_family={frontend_family}")
    solver = {
        "ML_ISTA_NET": "ista",
        "ML_LISTA_NET": "lista",
        "ML_FISTA_NET": "fista",
        "LBP_NET": "lbp",
    }.get(solver_type, solver_type)
    if solver not in {"ista", "lista", "fista", "lbp"}:
        raise ValueError(f"Unsupported solver_type={solver_type}")
    unroll_steps = int(lbp_unroll_steps if solver == "lbp" else lista_unroll_steps)
    spec = FRONTEND_SPECS[frontend_family]
    sector_sizes = [len(group) for group in bundle.sector_groups]
    if frontend_family == "FS_sl_hsa_l2":
        return SignatureSectorAttentionPaperSweepModel(
            sector_sizes=sector_sizes,
            n_assets=int(bundle.meta["n_assets"]),
            output_len=int(bundle.y_train_reg.shape[1]),
            features_per_asset=int(bundle.meta["features_per_asset"]),
            spec=spec,
            solver_type=solver,
            embed_dim=32,
            num_heads=num_heads,
            unroll_steps=unroll_steps,
            dropout_rate=dropout_rate,
        )
    return OHLCGRUPaperSweepModel(
        sector_sizes=sector_sizes,
        n_assets=int(bundle.meta["n_assets"]),
        output_len=int(bundle.y_train_reg.shape[1]),
        seq_len=int(bundle.meta["seq_len"]),
        features_per_asset=int(bundle.meta["features_per_asset"]),
        spec=spec,
        solver_type=solver,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        n_attention_layers=n_attention_layers,
        unroll_steps=unroll_steps,
        dropout_rate=dropout_rate,
    )
