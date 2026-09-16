from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_model import AttentionBlock, SequenceBatchNorm


def _same_padding(size: int, kernel: int, stride: int) -> tuple[int, int]:
    output = math.ceil(size / stride)
    total = max((output - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


def _conv_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
) -> torch.Tensor:
    pad_top, pad_bottom = _same_padding(x.shape[-2], weight.shape[-2], stride[0])
    pad_left, pad_right = _same_padding(x.shape[-1], weight.shape[-1], stride[1])
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return F.conv2d(padded, weight, stride=stride)


def _deconv_to_shape(
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    *,
    stride: tuple[int, int],
) -> torch.Tensor:
    target_h, target_w = reference.shape[-2:]
    raw_h = (x.shape[-2] - 1) * stride[0] + weight.shape[-2]
    raw_w = (x.shape[-1] - 1) * stride[1] + weight.shape[-1]
    output_padding = (
        max(min(target_h - raw_h, stride[0] - 1), 0),
        max(min(target_w - raw_w, stride[1] - 1), 0),
    )
    y = F.conv_transpose2d(
        x,
        weight,
        stride=stride,
        output_padding=output_padding,
    )
    if y.shape[-2] < target_h or y.shape[-1] < target_w:
        y = F.pad(
            y,
            (0, max(target_w - y.shape[-1], 0), 0, max(target_h - y.shape[-2], 0)),
        )
    return y[..., :target_h, :target_w]


class AssetAlignedMLLISTA(nn.Module):
    """LISTA whose first layer maps one complete asset feature block to one token."""

    def __init__(
        self,
        *,
        seq_len: int,
        features_per_asset: int,
        time_kernel: int,
        channels: int = 32,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        if seq_len < 1 or features_per_asset < 1:
            raise ValueError("seq_len and features_per_asset must be positive")
        if time_kernel < 1 or time_kernel > seq_len:
            raise ValueError(f"time_kernel={time_kernel} is invalid for seq_len={seq_len}")

        self.seq_len = int(seq_len)
        self.features_per_asset = int(features_per_asset)
        self.time_kernel = int(time_kernel)
        self.unroll_steps = int(unroll_steps)
        self.first_stride = (1, self.features_per_asset)

        first_kernel = (self.time_kernel, self.features_per_asset)
        latent_kernel = (1, 1) if self.seq_len == 1 else (2, 1)
        self.B1 = nn.Parameter(torch.empty(channels, 1, *first_kernel))
        self.B2 = nn.Parameter(torch.empty(channels, channels, *latent_kernel))
        self.B3 = nn.Parameter(torch.empty(channels, channels, *latent_kernel))
        self.W1 = nn.Parameter(torch.empty(channels, 1, *first_kernel))
        self.W2 = nn.Parameter(torch.empty(channels, channels, *latent_kernel))
        self.W3 = nn.Parameter(torch.empty(channels, channels, *latent_kernel))
        self.bias1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._reset_parameters(channels)

    def _reset_parameters(self, channels: int) -> None:
        nn.init.normal_(self.B1, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.W1, mean=0.0, std=0.1 / np.sqrt(36))
        nn.init.normal_(self.B2, mean=0.0, std=0.1 / np.sqrt(channels * 36))
        nn.init.normal_(self.W2, mean=0.0, std=0.1 / np.sqrt(channels * 36))
        nn.init.normal_(self.B3, mean=0.0, std=0.1 / np.sqrt(channels * 16))
        nn.init.normal_(self.W3, mean=0.0, std=0.1 / np.sqrt(channels * 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,A*F), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got {x.shape[1]}")
        if x.shape[2] % self.features_per_asset:
            raise ValueError(
                f"Width {x.shape[2]} is not divisible by F={self.features_per_asset}"
            )
        n_assets = x.shape[2] // self.features_per_asset
        x_img = x.unsqueeze(1)

        gamma1 = torch.relu(
            _conv_same(x_img, self.B1, stride=self.first_stride) + self.bias1
        )
        if gamma1.shape[-1] != n_assets:
            raise RuntimeError(
                f"Asset alignment failed: expected width {n_assets}, got {gamma1.shape[-1]}"
            )
        gamma2 = torch.relu(_conv_same(gamma1, self.B2, stride=(1, 1)) + self.bias2)
        gamma3 = torch.relu(_conv_same(gamma2, self.B3, stride=(1, 1)) + self.bias3)

        for _ in range(self.unroll_steps):
            gamma2_back = _deconv_to_shape(
                gamma3, self.B3, gamma2, stride=(1, 1)
            )
            gamma1_back = _deconv_to_shape(
                gamma2_back, self.B2, gamma1, stride=(1, 1)
            )
            reconstructed_x = _deconv_to_shape(
                gamma1_back, self.W1, x_img, stride=self.first_stride
            )
            gamma1 = torch.relu(
                gamma1_back
                - _conv_same(reconstructed_x, self.W1, stride=self.first_stride)
                + _conv_same(x_img, self.B1, stride=self.first_stride)
                + self.bias1
            )
            gamma2_reconstruction = _deconv_to_shape(
                gamma2_back, self.W2, gamma1, stride=(1, 1)
            )
            gamma2 = torch.relu(
                gamma2_back
                - _conv_same(gamma2_reconstruction, self.W2, stride=(1, 1))
                + _conv_same(gamma1, self.B2, stride=(1, 1))
                + self.bias2
            )
            gamma3_reconstruction = _deconv_to_shape(
                gamma3, self.W3, gamma2, stride=(1, 1)
            )
            gamma3 = torch.relu(
                gamma3
                - _conv_same(gamma3_reconstruction, self.W3, stride=(1, 1))
                + _conv_same(gamma2, self.B3, stride=(1, 1))
                + self.bias3
            )

        # Preserve both time and asset axes for the downstream architecture.
        return gamma3.permute(0, 2, 3, 1).contiguous()  # (B,T,A,32)


class AssetAlignedTemporalSector(nn.Module):
    def __init__(
        self,
        *,
        sector_size: int,
        seq_len: int,
        features_per_asset: int,
        hidden_dim: int = 64,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = AssetAlignedMLLISTA(
            seq_len=seq_len,
            features_per_asset=features_per_asset,
            time_kernel=4,
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


class AssetAlignedSignatureSector(nn.Module):
    def __init__(
        self,
        *,
        sector_size: int,
        features_per_asset: int,
        hidden_dim: int = 64,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = AssetAlignedMLLISTA(
            seq_len=1,
            features_per_asset=features_per_asset,
            time_kernel=1,
            unroll_steps=unroll_steps,
        )
        self.projection = nn.Linear(int(sector_size) * 32, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_assets = self.frontend(x)
        sparse_sector = sparse_assets.flatten(start_dim=2)
        sector_token = self.dropout(self.act(self.norm(self.projection(sparse_sector))))
        return sector_token, sparse_sector


class AssetAlignedSignatureBlockSector(nn.Module):
    def __init__(
        self,
        *,
        sector_size: int,
        seq_len: int,
        raw_features_per_asset: int,
        hidden_dim: int = 64,
        unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if raw_features_per_asset != 24:
            raise ValueError(
                f"Expected OHLC4 + Signature20, got {raw_features_per_asset}"
            )
        self.sector_size = int(sector_size)
        self.signature_norm = nn.LayerNorm(20)
        self.signature_projection = nn.Linear(20, 4)
        self.signature_post_norm = nn.LayerNorm(4)
        self.frontend = AssetAlignedMLLISTA(
            seq_len=seq_len,
            features_per_asset=8,
            time_kernel=4,
            unroll_steps=unroll_steps,
        )
        self.gru = nn.GRU(self.sector_size * 32, hidden_dim, batch_first=True)
        self.bn = SequenceBatchNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        assets = x.reshape(batch, seq_len, self.sector_size, 24)
        ohlc = assets[..., :4]
        signature = self.signature_post_norm(
            self.signature_projection(self.signature_norm(assets[..., 4:]))
        )
        adapted = torch.cat([ohlc, signature], dim=-1).reshape(
            batch, seq_len, self.sector_size * 8
        )
        sparse_assets = self.frontend(adapted)
        sparse_sector = sparse_assets.flatten(start_dim=2)
        encoded, _ = self.gru(sparse_sector)
        encoded = self.dropout(self.act(self.bn(encoded)))
        return encoded, sparse_sector


class AssetAlignedProposedModel(nn.Module):
    def __init__(
        self,
        *,
        sector_sizes: List[int],
        n_assets: int,
        output_len: int,
        seq_len: int,
        features_per_asset: int,
        architecture: str,
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_attention_layers: int = 3,
        lista_unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture not in {"ohlc_gru", "signature_linear", "signature_block_gru"}:
            raise ValueError(f"Unsupported architecture={architecture}")
        self.architecture = architecture
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)
        encoders = []
        for sector_size in sector_sizes:
            common = {
                "sector_size": sector_size,
                "hidden_dim": hidden_dim,
                "unroll_steps": lista_unroll_steps,
                "dropout_rate": dropout_rate,
            }
            if architecture == "ohlc_gru":
                encoder = AssetAlignedTemporalSector(
                    seq_len=seq_len,
                    features_per_asset=features_per_asset,
                    **common,
                )
            elif architecture == "signature_linear":
                encoder = AssetAlignedSignatureSector(
                    features_per_asset=features_per_asset,
                    **common,
                )
            else:
                encoder = AssetAlignedSignatureBlockSector(
                    seq_len=seq_len,
                    raw_features_per_asset=features_per_asset,
                    **common,
                )
            encoders.append(encoder)
        self.sector_encoders = nn.ModuleList(encoders)
        concat_dim = hidden_dim * len(sector_sizes)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(concat_dim, num_heads, dropout_rate=dropout_rate)
                for _ in range(n_attention_layers)
            ]
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
            [
                head(flat).reshape(x.shape[0], self.output_len, 1, 2)
                for head in self.cls_heads
            ],
            dim=2,
        )
        return {
            "reg_out": reg_out,
            "cls_logits": cls_logits,
            "cls_out": torch.softmax(cls_logits, dim=-1),
            "con_out": torch.cat(sparse, dim=-1),
        }


class AssetAlignedSignatureAssetAttentionModel(nn.Module):
    """Full-window Signature LISTA followed directly by cross-asset attention."""

    def __init__(
        self,
        *,
        sector_sizes: List[int],
        n_assets: int,
        output_len: int,
        features_per_asset: int,
        num_heads: int = 4,
        n_attention_layers: int = 3,
        lista_unroll_steps: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.sector_sizes = list(sector_sizes)
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)
        self.frontends = nn.ModuleList(
            [
                AssetAlignedMLLISTA(
                    seq_len=1,
                    features_per_asset=features_per_asset,
                    time_kernel=1,
                    unroll_steps=lista_unroll_steps,
                )
                for _ in sector_sizes
            ]
        )
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(32, num_heads, dropout_rate=dropout_rate)
                for _ in range(n_attention_layers)
            ]
        )
        self.head_dropout = nn.Dropout(dropout_rate)
        flat_dim = self.n_assets * 32
        self.reg_head = nn.Linear(flat_dim, self.output_len * self.n_assets)
        self.cls_heads = nn.ModuleList(
            [nn.Linear(flat_dim, self.output_len * 2) for _ in range(self.n_assets)]
        )

    def forward(self, x_sector_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = [
            frontend(x_sector)[:, 0, :, :]
            for frontend, x_sector in zip(self.frontends, x_sector_list)
        ]
        x = torch.cat(tokens, dim=1)
        if x.shape[1] != self.n_assets:
            raise RuntimeError(f"Expected {self.n_assets} asset tokens, got {x.shape[1]}")
        for block in self.attention_blocks:
            x = block(x)
        flat = self.head_dropout(x.flatten(start_dim=1))
        reg_out = self.reg_head(flat).reshape(x.shape[0], self.output_len, self.n_assets)
        cls_logits = torch.cat(
            [
                head(flat).reshape(x.shape[0], self.output_len, 1, 2)
                for head in self.cls_heads
            ],
            dim=2,
        )
        return {
            "reg_out": reg_out,
            "cls_logits": cls_logits,
            "cls_out": torch.softmax(cls_logits, dim=-1),
            "con_out": x,
        }


def build_asset_aligned_model_from_data(
    bundle,
    *,
    architecture: str,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_attention_layers: int = 3,
    lista_unroll_steps: int = 4,
    dropout_rate: float = 0.1,
    **_: object,
) -> nn.Module:
    if architecture == "signature_asset_attention":
        return AssetAlignedSignatureAssetAttentionModel(
            sector_sizes=[len(group) for group in bundle.sector_groups],
            n_assets=int(bundle.meta["n_assets"]),
            output_len=int(bundle.y_train_reg.shape[1]),
            features_per_asset=int(bundle.meta["features_per_asset"]),
            num_heads=num_heads,
            n_attention_layers=n_attention_layers,
            lista_unroll_steps=lista_unroll_steps,
            dropout_rate=dropout_rate,
        )
    return AssetAlignedProposedModel(
        sector_sizes=[len(group) for group in bundle.sector_groups],
        n_assets=int(bundle.meta["n_assets"]),
        output_len=int(bundle.y_train_reg.shape[1]),
        seq_len=int(bundle.meta["seq_len"]),
        features_per_asset=int(bundle.meta["features_per_asset"]),
        architecture=architecture,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        n_attention_layers=n_attention_layers,
        lista_unroll_steps=lista_unroll_steps,
        dropout_rate=dropout_rate,
    )
