from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from torch_model import AttentionBlock, SequenceBatchNorm


class NoSparseSectorGRUEncoder(nn.Module):
    """Sector-wise raw OHLC encoder used for the no-sparse frontend ablation.

    Input per sector is already grouped as (B, T, A_sector * F).  This follows
    the CSC+UAR no_feature_extractor path: sector split and normalization happen
    before the predictor, but no CSC/LISTA sparse transform is applied.
    """

    def __init__(
        self,
        *,
        sector_size: int,
        features_per_asset: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.sector_size = int(sector_size)
        self.features_per_asset = int(features_per_asset)
        self.input_dim = self.sector_size * self.features_per_asset
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.bn = SequenceBatchNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected sector input (B,T,A_s*F), got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected sector width {self.input_dim}, got {x.shape[-1]}")
        encoded, _ = self.gru(x)
        encoded = self.dropout(self.act(self.bn(encoded)))
        return encoded, x


class NoSparseGRUAttentionModel(nn.Module):
    """No-sparse ablation: raw sector OHLC -> sector GRU -> attention -> heads."""

    def __init__(
        self,
        *,
        sector_sizes: List[int],
        n_assets: int,
        output_len: int,
        features_per_asset: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_attention_layers: int = 3,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.output_len = int(output_len)
        self.features_per_asset = int(features_per_asset)
        self.sector_encoders = nn.ModuleList(
            [
                NoSparseSectorGRUEncoder(
                    sector_size=size,
                    features_per_asset=self.features_per_asset,
                    hidden_dim=hidden_dim,
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
        encoded, raw_sectors = [], []
        for encoder, x_sector in zip(self.sector_encoders, x_sector_list):
            sector_encoded, sector_raw = encoder(x_sector)
            encoded.append(sector_encoded)
            raw_sectors.append(sector_raw)
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
            "con_out": torch.cat(raw_sectors, dim=-1),
        }


def build_nosparse_gru_attention_model_from_data(
    bundle,
    *,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_attention_layers: int = 3,
    dropout_rate: float = 0.1,
    **_: object,
) -> nn.Module:
    return NoSparseGRUAttentionModel(
        sector_sizes=[len(group) for group in bundle.sector_groups],
        n_assets=int(bundle.meta["n_assets"]),
        output_len=int(bundle.y_train_reg.shape[1]),
        features_per_asset=int(bundle.meta["features_per_asset"]),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        n_attention_layers=n_attention_layers,
        dropout_rate=dropout_rate,
    )
