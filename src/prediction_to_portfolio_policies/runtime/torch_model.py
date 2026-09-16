from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_data import ProposedDataBundle, load_proposed_djia_data


class SequenceBatchNorm(nn.Module):
    """BatchNorm1d for sequence tensors shaped as (B, T, C)."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SequenceBatchNorm expects (B,T,C), got {tuple(x.shape)}")
        x_bn = x.transpose(1, 2)
        # Online signature runs can have B=1 and T=1. PyTorch BatchNorm cannot
        # estimate batch statistics from a single value per channel, so use the
        # stored running statistics in that degenerate short-update case.
        values_per_channel = x_bn.shape[0] * x_bn.shape[2]
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                x_bn,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                eps=self.bn.eps,
            ).transpose(1, 2)
        return self.bn(x_bn).transpose(1, 2)


class SectorEncoder(nn.Module):
    """Dense frontend + GRU block used by the active TensorFlow proposed model."""

    def __init__(self, input_dim: int, frontend_dim: int = 32, hidden_dim: int = 64, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.frontend = nn.Linear(input_dim, frontend_dim)
        self.gru = nn.GRU(input_size=frontend_dim, hidden_size=hidden_dim, batch_first=True)
        self.bn = SequenceBatchNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        con = self.frontend(x)
        x, _ = self.gru(con)
        x = self.bn(x)
        return self.dropout(self.act(x)), con


def _same_pad_2d(x: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    pad_h = max(kernel_size[0] - 1, 0)
    pad_w = max(kernel_size[1] - 1, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom))


class TorchMLLISTA(nn.Module):
    """Torch port of the active ML_LISTA_NET frontend.

    Input shape is (B, T, sector_assets * features_per_asset). Output remains
    sequence-shaped as (B, T, encoded_dim), matching the TensorFlow reshape.
    """

    def __init__(
        self,
        h_dim: int,
        m0: int,
        m1: int = 32,
        m2: int = 32,
        m3: int = 32,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        self.h_dim = int(h_dim)
        self.m0 = int(m0)
        self.unroll_steps = int(unroll_steps)

        self.B1 = nn.Conv2d(1, m1, kernel_size=(4, self.m0), padding=0, bias=False)
        self.B2 = nn.Conv2d(m1, m2, kernel_size=(2, 2), padding=0, bias=False)
        self.B3 = nn.Conv2d(m2, m3, kernel_size=(2, 2), padding=0, bias=False)
        self.W1 = nn.Conv2d(1, m1, kernel_size=(2, 2), padding=0, bias=False)
        self.W2 = nn.Conv2d(m1, m2, kernel_size=(2, 2), padding=0, bias=False)
        self.W3 = nn.Conv2d(m2, m3, kernel_size=(2, 2), padding=0, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1, m1, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, m2, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, m3, 1, 1))
        self._reset_parameters(m1=m1, m2=m2)

    def _reset_parameters(self, m1: int, m2: int) -> None:
        nn.init.normal_(self.B1.weight, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.B2.weight, mean=0.0, std=0.1 / np.sqrt(m1 * 36))
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.1 / np.sqrt(m1 * 36))
        nn.init.normal_(self.B3.weight, mean=0.0, std=0.1 / np.sqrt(m2 * 16))
        nn.init.normal_(self.W3.weight, mean=0.0, std=0.1 / np.sqrt(m2 * 16))

    @staticmethod
    def _conv_same(layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return layer(_same_pad_2d(x, layer.kernel_size))

    @staticmethod
    def _tied_deconv_same(layer: nn.Conv2d, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(x, layer.weight, bias=None, stride=1)
        if y.shape[-2] < ref.shape[-2] or y.shape[-1] < ref.shape[-1]:
            pad_h = max(ref.shape[-2] - y.shape[-2], 0)
            pad_w = max(ref.shape[-1] - y.shape[-1], 0)
            y = F.pad(y, (0, pad_w, 0, pad_h))
        if y.shape[-2:] != ref.shape[-2:]:
            y = y[..., : ref.shape[-2], : ref.shape[-1]]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TorchMLLISTA expects (B,T,F), got {tuple(x.shape)}")
        x_img = x.unsqueeze(1)

        gamma1 = torch.relu(self._conv_same(self.B1, x_img) + self.bias1)
        gamma2 = torch.relu(self._conv_same(self.B2, gamma1) + self.bias2)
        gamma3 = torch.relu(self._conv_same(self.B3, gamma2) + self.bias3)

        for _ in range(self.unroll_steps):
            gamma2_back = self._tied_deconv_same(self.B3, gamma3, gamma2)
            gamma1_back = self._tied_deconv_same(self.B2, gamma2_back, gamma1)
            gamma1_residual = self._tied_deconv_same(self.W1, gamma1_back, x_img)
            gamma1 = torch.relu(
                gamma1_back
                - self._conv_same(self.W1, gamma1_residual)
                + self._conv_same(self.B1, x_img)
                + self.bias1
            )

            gamma2_residual = self._tied_deconv_same(self.W2, gamma2_back, gamma1)
            gamma2 = torch.relu(
                gamma2_back
                - self._conv_same(self.W2, gamma2_residual)
                + self._conv_same(self.B2, gamma1)
                + self.bias2
            )

            gamma3_residual = self._tied_deconv_same(self.W3, gamma3, gamma2)
            gamma3 = torch.relu(
                gamma3
                - self._conv_same(self.W3, gamma3_residual)
                + self._conv_same(self.B3, gamma2)
                + self.bias3
            )

        # (B, C, T, W) -> (B, T, W*C), equivalent to Keras Reshape((h_dim, -1)).
        gamma3 = gamma3.permute(0, 2, 3, 1).contiguous()
        return gamma3.reshape(gamma3.shape[0], gamma3.shape[1], -1)


class TorchMLISTANet(nn.Module):
    """Torch port of ML_ISTA_NET with tied convolution/deconvolution kernels."""

    def __init__(
        self,
        h_dim: int,
        m0: int,
        m1: int = 32,
        m2: int = 32,
        m3: int = 32,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        self.h_dim = int(h_dim)
        self.m0 = int(m0)
        self.unroll_steps = int(unroll_steps)

        self.W1 = nn.Conv2d(1, m1, kernel_size=(4, self.m0), padding=0, bias=False)
        self.W2 = nn.Conv2d(m1, m2, kernel_size=(2, 2), padding=0, bias=False)
        self.W3 = nn.Conv2d(m2, m3, kernel_size=(2, 2), padding=0, bias=False)
        self.c1 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.c2 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.c3 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.bias1 = nn.Parameter(torch.zeros(1, m1, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, m2, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, m3, 1, 1))
        # Backward-compatible aliases for TorchMLFISTANet, which inherits this class.
        self.w1 = self.W1
        self.w2 = self.W2
        self.w3 = self.W3
        self.b1 = self.bias1
        self.b2 = self.bias2
        self.b3 = self.bias3
        self._reset_parameters(m1=m1, m2=m2)

    def _reset_parameters(self, m1: int, m2: int) -> None:
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.1 / np.sqrt(m1 * 36))
        nn.init.normal_(self.W3.weight, mean=0.0, std=0.1 / np.sqrt(m2 * 16))

    @staticmethod
    def _conv_same(layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return layer(_same_pad_2d(x, layer.kernel_size))

    @staticmethod
    def _tied_deconv_same(layer: nn.Conv2d, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(x, layer.weight, bias=None, stride=1)
        if y.shape[-2] < ref.shape[-2] or y.shape[-1] < ref.shape[-1]:
            pad_h = max(ref.shape[-2] - y.shape[-2], 0)
            pad_w = max(ref.shape[-1] - y.shape[-1], 0)
            y = F.pad(y, (0, pad_w, 0, pad_h))
        if y.shape[-2:] != ref.shape[-2:]:
            y = y[..., : ref.shape[-2], : ref.shape[-1]]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TorchMLISTANet expects (B,T,F), got {tuple(x.shape)}")
        x_img = x.unsqueeze(1)

        gamma1 = torch.relu(self.c1 * self._conv_same(self.W1, x_img) + self.bias1)
        gamma2 = torch.relu(self.c2 * self._conv_same(self.W2, gamma1) + self.bias2)
        gamma3 = torch.relu(self.c3 * self._conv_same(self.W3, gamma2) + self.bias3)

        for _ in range(self.unroll_steps):
            gamma2 = self._tied_deconv_same(self.W3, gamma3, gamma2)
            gamma1 = self._tied_deconv_same(self.W2, gamma2, gamma1)
            gamma1_residual = self._tied_deconv_same(self.W1, gamma1, x_img) - x_img
            gamma1 = torch.relu(
                gamma1 - self.c1 * self._conv_same(self.W1, gamma1_residual) + self.bias1
            )

            gamma2_residual = self._tied_deconv_same(self.W2, gamma2, gamma1) - gamma1
            gamma2 = torch.relu(
                gamma2 - self.c2 * self._conv_same(self.W2, gamma2_residual) + self.bias2
            )

            gamma3_residual = self._tied_deconv_same(self.W3, gamma3, gamma2) - gamma2
            gamma3 = torch.relu(
                gamma3 - self.c3 * self._conv_same(self.W3, gamma3_residual) + self.bias3
            )

        gamma3 = gamma3.permute(0, 2, 3, 1).contiguous()
        return gamma3.reshape(gamma3.shape[0], gamma3.shape[1], -1)


class TorchMLFISTANet(TorchMLISTANet):
    """Torch port of ML_FISTA_NET, adding Nesterov acceleration on gamma3."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TorchMLFISTANet expects (B,T,F), got {tuple(x.shape)}")
        x_img = x.unsqueeze(1)

        gamma1 = torch.relu(self.c1 * self._conv_same(self.W1, x_img) + self.bias1)
        gamma2 = torch.relu(self.c2 * self._conv_same(self.W2, gamma1) + self.bias2)
        gamma3 = torch.relu(self.c3 * self._conv_same(self.W3, gamma2) + self.bias3)
        gamma3_prev = gamma3
        t = 1.0

        for _ in range(self.unroll_steps):
            t_prev = t
            t = (1.0 + float(np.sqrt(1.0 + 4.0 * t_prev**2))) / 2.0
            z = gamma3 + ((t_prev - 1.0) / t) * (gamma3 - gamma3_prev)
            gamma3_prev = gamma3

            gamma2_back = self._tied_deconv_same(self.W3, z, gamma2)
            gamma1_back = self._tied_deconv_same(self.W2, gamma2_back, gamma1)
            gamma1_residual = self._tied_deconv_same(self.W1, gamma1_back, x_img) - x_img
            gamma1 = torch.relu(
                gamma1_back - self.c1 * self._conv_same(self.W1, gamma1_residual) + self.bias1
            )

            gamma2_residual = self._tied_deconv_same(self.W2, gamma2_back, gamma1) - gamma1
            gamma2 = torch.relu(
                gamma2_back - self.c2 * self._conv_same(self.W2, gamma2_residual) + self.bias2
            )

            gamma3_residual = self._tied_deconv_same(self.W3, z, gamma2) - gamma2
            gamma3 = torch.relu(
                z - self.c3 * self._conv_same(self.W3, gamma3_residual) + self.bias3
            )

        gamma3 = gamma3.permute(0, 2, 3, 1).contiguous()
        return gamma3.reshape(gamma3.shape[0], gamma3.shape[1], -1)


class TorchLBPNet(nn.Module):
    """Torch port of LBP_NET frontend."""

    def __init__(
        self,
        h_dim: int,
        m0: int,
        m1: int = 32,
        m2: int = 32,
        m3: int = 32,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        self.h_dim = int(h_dim)
        self.m0 = int(m0)
        self.unroll_steps = int(unroll_steps)

        self.W1 = nn.Conv2d(1, m1, kernel_size=(4, self.m0), padding=0, bias=False)
        self.W2 = nn.Conv2d(m1, m2, kernel_size=(2, 2), padding=0, bias=False)
        self.W3 = nn.Conv2d(m2, m3, kernel_size=(2, 2), padding=0, bias=False)
        self.c1 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.c2 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.c3 = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.bias1 = nn.Parameter(torch.zeros(1, m1, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, m2, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, m3, 1, 1))
        self._reset_parameters(m1=m1, m2=m2)

    def _reset_parameters(self, m1: int, m2: int) -> None:
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.1 / np.sqrt(m1 * 36))
        nn.init.normal_(self.W3.weight, mean=0.0, std=0.1 / np.sqrt(m2 * 16))

    @staticmethod
    def _conv_same(layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return layer(_same_pad_2d(x, layer.kernel_size))

    @staticmethod
    def _tied_deconv_same(layer: nn.Conv2d, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(x, layer.weight, bias=None, stride=1)
        if y.shape[-2] < ref.shape[-2] or y.shape[-1] < ref.shape[-1]:
            pad_h = max(ref.shape[-2] - y.shape[-2], 0)
            pad_w = max(ref.shape[-1] - y.shape[-1], 0)
            y = F.pad(y, (0, pad_w, 0, pad_h))
        if y.shape[-2:] != ref.shape[-2:]:
            y = y[..., : ref.shape[-2], : ref.shape[-1]]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TorchLBPNet expects (B,T,F), got {tuple(x.shape)}")
        x_img = x.unsqueeze(1)

        gamma1 = torch.relu(self.c1 * self._conv_same(self.W1, x_img) + self.bias1)
        if self.unroll_steps > 0:
            for _ in range(self.unroll_steps):
                residual1 = self._tied_deconv_same(self.W1, gamma1, x_img) - x_img
                gamma1 = torch.relu(gamma1 - self.c1 * self._conv_same(self.W1, residual1) + self.bias1)

        gamma2 = torch.relu(self.c2 * self._conv_same(self.W2, gamma1) + self.bias2)
        if self.unroll_steps > 0:
            for _ in range(self.unroll_steps):
                residual2 = self._tied_deconv_same(self.W2, gamma2, gamma1) - gamma1
                gamma2 = torch.relu(gamma2 - self.c2 * self._conv_same(self.W2, residual2) + self.bias2)

        gamma3 = torch.relu(self.c3 * self._conv_same(self.W3, gamma2) + self.bias3)
        if self.unroll_steps > 0:
            for _ in range(self.unroll_steps):
                residual3 = self._tied_deconv_same(self.W3, gamma3, gamma2) - gamma2
                gamma3 = torch.relu(gamma3 - self.c3 * self._conv_same(self.W3, residual3) + self.bias3)

        gamma3 = gamma3.permute(0, 2, 3, 1).contiguous()
        return gamma3.reshape(gamma3.shape[0], gamma3.shape[1], -1)


class SparseSectorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        features_per_asset: int,
        hidden_dim: int = 64,
        sparse_type: str = "lista",
        unroll_steps: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim % features_per_asset != 0:
            raise ValueError(f"input_dim={input_dim} is not divisible by features_per_asset={features_per_asset}")
        if sparse_type in {"lbp", "LBP_NET"}:
            frontend_cls = TorchLBPNet
        elif sparse_type in {"ista", "ML_ISTA_NET"}:
            frontend_cls = TorchMLISTANet
        elif sparse_type in {"fista", "ML_FISTA_NET"}:
            frontend_cls = TorchMLFISTANet
        else:
            frontend_cls = TorchMLLISTA
        self.frontend = frontend_cls(
            h_dim=seq_len, m0=features_per_asset, m1=32, m2=32, m3=32, unroll_steps=unroll_steps
        )
        self.gru: nn.GRU | None = None
        self.hidden_dim = int(hidden_dim)
        self.bn = SequenceBatchNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0 else nn.Identity()

    def _build_gru_if_needed(self, x: torch.Tensor) -> None:
        input_size = int(x.shape[-1])
        if self.gru is None:
            self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_dim, batch_first=True).to(x.device)
        elif self.gru.input_size != input_size:
            raise ValueError(f"LISTA encoded dim changed from {self.gru.input_size} to {input_size}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        con = self.frontend(x)
        self._build_gru_if_needed(con)
        if self.gru is None:
            raise RuntimeError("LISTA GRU was not initialized")
        x, _ = self.gru(con)
        x = self.bn(x)
        return self.dropout(self.act(x)), con


class AttentionBlock(nn.Module):
    """Self-attention + BatchNorm + ReLU block."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.bn = SequenceBatchNorm(embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.bn(attn_out)
        return self.dropout(self.act(attn_out))


class ProposedTorchModel(nn.Module):
    """PyTorch forward-pass equivalent of the proposed TensorFlow model.

    This first version intentionally focuses on the active dense-frontend path:
    five sector tensors -> sector encoders -> 3 attention blocks -> reg/cls heads.
    """

    def __init__(
        self,
        sector_input_dims: List[int],
        n_assets: int = 30,
        output_len: int = 1,
        frontend_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_attention_layers: int = 3,
        frontend_type: str = "dense",
        seq_len: int = 15,
        features_per_asset: int = 4,
        lista_unroll_steps: int = 4,
        lbp_unroll_steps: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(sector_input_dims) != 5:
            raise ValueError(f"Expected 5 sector dims, got {sector_input_dims}")
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)

        self.frontend_type = frontend_type
        if frontend_type == "dense":
            self.sector_encoders = nn.ModuleList(
                [
                    SectorEncoder(
                        dim,
                        frontend_dim=frontend_dim,
                        hidden_dim=hidden_dim,
                        dropout_rate=dropout_rate,
                    )
                    for dim in sector_input_dims
                ]
            )
        elif frontend_type in {"lista", "ML_LISTA_NET", "ista", "ML_ISTA_NET", "fista", "ML_FISTA_NET"}:
            sparse_type = {
                "ista": "ista",
                "ML_ISTA_NET": "ML_ISTA_NET",
                "fista": "fista",
                "ML_FISTA_NET": "ML_FISTA_NET",
            }.get(frontend_type, "lista")
            self.sector_encoders = nn.ModuleList(
                [
                    SparseSectorEncoder(
                        dim,
                        seq_len=seq_len,
                        features_per_asset=features_per_asset,
                        hidden_dim=hidden_dim,
                        sparse_type=sparse_type,
                        unroll_steps=lista_unroll_steps,
                        dropout_rate=dropout_rate,
                    )
                    for dim in sector_input_dims
                ]
            )
        elif frontend_type in {"lbp", "LBP_NET"}:
            self.sector_encoders = nn.ModuleList(
                [
                    SparseSectorEncoder(
                        dim,
                        seq_len=seq_len,
                        features_per_asset=features_per_asset,
                        hidden_dim=hidden_dim,
                        sparse_type="lbp",
                        unroll_steps=lbp_unroll_steps,
                        dropout_rate=dropout_rate,
                    )
                    for dim in sector_input_dims
                ]
            )
        else:
            raise ValueError(f"Unsupported frontend_type={frontend_type}")

        concat_dim = hidden_dim * len(sector_input_dims)
        if concat_dim % num_heads != 0:
            raise ValueError(f"concat_dim={concat_dim} must be divisible by num_heads={num_heads}")
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(concat_dim, num_heads=num_heads, dropout_rate=dropout_rate) for _ in range(n_attention_layers)]
        )
        self.head_dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0 else nn.Identity()

        self.flatten_dim: int | None = None
        self.reg_head: nn.Linear | None = None
        self.cls_heads = nn.ModuleList()

    def _build_heads_if_needed(self, x: torch.Tensor) -> None:
        flat_dim = x.shape[1] * x.shape[2]
        if self.flatten_dim == flat_dim:
            return
        if self.flatten_dim is not None and self.flatten_dim != flat_dim:
            raise ValueError(f"Model was initialized for flat_dim={self.flatten_dim}, got {flat_dim}")

        self.flatten_dim = int(flat_dim)
        self.reg_head = nn.Linear(flat_dim, self.output_len * self.n_assets).to(x.device)
        self.cls_heads = nn.ModuleList(
            [nn.Linear(flat_dim, self.output_len * 2).to(x.device) for _ in range(self.n_assets)]
        )

    def forward(self, x_sector_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(x_sector_list) != len(self.sector_encoders):
            raise ValueError(f"Expected {len(self.sector_encoders)} sector inputs, got {len(x_sector_list)}")

        encoded = []
        con_parts = []
        for encoder, x_in in zip(self.sector_encoders, x_sector_list):
            enc, con = encoder(x_in)
            encoded.append(enc)
            con_parts.append(con)
        con_out = torch.cat(con_parts, dim=-1)
        x = torch.cat(encoded, dim=-1)
        for block in self.attention_blocks:
            x = block(x)

        self._build_heads_if_needed(x)
        flat = self.head_dropout(x.reshape(x.shape[0], -1))

        if self.reg_head is None:
            raise RuntimeError("reg_head was not initialized")
        reg_out = self.reg_head(flat).reshape(x.shape[0], self.output_len, self.n_assets)

        cls_logits = []
        for head in self.cls_heads:
            cls_logits.append(head(flat).reshape(x.shape[0], self.output_len, 1, 2))
        cls_logits_t = torch.cat(cls_logits, dim=2)
        cls_prob = torch.softmax(cls_logits_t, dim=-1)

        return {
            "reg_out": reg_out,
            "cls_logits": cls_logits_t,
            "cls_out": cls_prob,
            "con_out": con_out,
        }


def build_model_from_data(
    bundle: ProposedDataBundle,
    frontend_dim: int = 32,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_attention_layers: int = 3,
    frontend_type: str = "dense",
    lista_unroll_steps: int = 4,
    lbp_unroll_steps: int = 4,
    dropout_rate: float = 0.0,
) -> ProposedTorchModel:
    sector_dims = [int(x.shape[-1]) for x in bundle.x_train_list]
    return ProposedTorchModel(
        sector_input_dims=sector_dims,
        n_assets=int(bundle.meta["n_assets"]),
        output_len=int(bundle.y_train_reg.shape[1]),
        frontend_dim=frontend_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        n_attention_layers=n_attention_layers,
        frontend_type=frontend_type,
        seq_len=int(bundle.meta["seq_len"]),
        features_per_asset=int(bundle.meta["features_per_asset"]),
        lista_unroll_steps=lista_unroll_steps,
        lbp_unroll_steps=lbp_unroll_steps,
        dropout_rate=dropout_rate,
    )


def run_forward_check(
    root: str | Path = ".",
    out: str | Path = "outputs/torch_proposed/model_check.json",
    batch_size: int = 8,
    device: str = "cpu",
    frontend_type: str = "dense",
    use_signature: bool = False,
    signature_level: int = 2,
) -> Dict[str, object]:
    bundle = load_proposed_djia_data(
        root=root,
        device=device,
        use_signature=use_signature,
        signature_level=signature_level,
    )
    model = build_model_from_data(bundle, frontend_type=frontend_type).to(device)
    model.train()

    x_batch = [x[:batch_size] for x in bundle.x_train_list]
    with torch.no_grad():
        outputs = model(x_batch)

    cls_sum = outputs["cls_out"].sum(dim=-1)
    expected_cls_sum = torch.ones_like(cls_sum)
    report: Dict[str, object] = {
        "meta": bundle.meta,
        "config": {
            "batch_size": int(batch_size),
            "device": str(device),
            "sector_input_dims": [int(x.shape[-1]) for x in bundle.x_train_list],
            "model_path": "torch_model.py",
            "frontend": "Dense(32)",
            "frontend_type": frontend_type,
            "use_signature": bool(use_signature),
            "signature_level": int(signature_level) if use_signature else None,
            "sector_encoder": "GRU(64)+BatchNorm+ReLU",
            "attention_layers": 3,
            "attention_heads": 4,
        },
        "shapes": {
            "reg_out": list(outputs["reg_out"].shape),
            "cls_logits": list(outputs["cls_logits"].shape),
            "cls_out": list(outputs["cls_out"].shape),
        },
        "checks": {
            "reg_out_shape_ok": list(outputs["reg_out"].shape) == [batch_size, 1, int(bundle.meta["n_assets"])],
            "cls_out_shape_ok": list(outputs["cls_out"].shape) == [batch_size, 1, int(bundle.meta["n_assets"]), 2],
            "cls_prob_sum_is_one": bool(torch.allclose(cls_sum, expected_cls_sum, atol=1e-5)),
            "reg_out_is_finite": bool(torch.isfinite(outputs["reg_out"]).all().item()),
            "cls_out_is_finite": bool(torch.isfinite(outputs["cls_out"]).all().item()),
        },
        "sample": {
            "reg_out_first_asset_first_5": [
                float(v) for v in outputs["reg_out"][0, 0, :5].detach().cpu().tolist()
            ],
            "cls_out_first_asset": [
                float(v) for v in outputs["cls_out"][0, 0, 0].detach().cpu().tolist()
            ],
        },
    }

    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = Path(root) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-pass check for proposed Torch model.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/torch_proposed/model_check.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--frontend_type",
        default="dense",
        choices=["dense", "lista", "ML_LISTA_NET", "ista", "ML_ISTA_NET", "fista", "ML_FISTA_NET", "lbp", "LBP_NET"],
    )
    parser.add_argument("--use_signature", action="store_true")
    parser.add_argument("--signature_level", type=int, default=2)
    args = parser.parse_args()

    report = run_forward_check(
        root=args.root,
        out=args.out,
        batch_size=args.batch_size,
        device=args.device,
        frontend_type=args.frontend_type,
        use_signature=args.use_signature,
        signature_level=args.signature_level,
    )
    print(json.dumps(report["checks"], indent=2))
    print(json.dumps(report["shapes"], indent=2))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(args.root) / out_path
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
