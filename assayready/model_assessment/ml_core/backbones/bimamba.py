from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data_core.config import DnaConfig
from .common import BiMamba, DropPath, RMSNorm


class FullBiMambaBlock(nn.Module):
    """Pre-norm bidirectional Mamba block with a dense SwiGLU FFN."""

    def __init__(self, config: DnaConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        ffn_dim = 4 * hidden_size

        self.norm_mixer = RMSNorm(hidden_size)
        self.mixer = BiMamba(
            d_model=hidden_size,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            use_mamba2=config.use_mamba2,
            headdim=config.mamba2_head_dim,
            ngroups=config.mamba2_ngroups,
            mamba_version=config.mamba_version,
        )

        self.norm_ffn = RMSNorm(hidden_size)
        self.w1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w2 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = residual + self.drop_path(self.mixer(self.norm_mixer(x)))

        residual = x
        x_norm = self.norm_ffn(x)
        x_ffn = self.dropout(self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm)))
        return residual + self.drop_path(x_ffn)


class FullBiMambaBackbone(nn.Module):
    """Full-resolution BiMamba stack for an isolated backbone probe."""

    def __init__(self, config: DnaConfig) -> None:
        super().__init__()
        self.config = config
        drop_path_rates = [
            item.item()
            for item in torch.linspace(0, config.drop_path_rate, config.high_level_layers)
        ]
        self.layers = nn.ModuleList(
            [
                FullBiMambaBlock(config, drop_path_rate=drop_path_rates[layer_idx])
                for layer_idx in range(config.high_level_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None, None]:
        del src_key_padding_mask
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                src = torch.utils.checkpoint.checkpoint(layer, src, use_reentrant=False)
            else:
                src = layer(src)
        return self.final_norm(src), None, None
