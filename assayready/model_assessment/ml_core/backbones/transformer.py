from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...data_core.config import DnaConfig
from .common import RMSNorm, RoPECache, TransformerEncoderLayerRoPE


class PlainTransformerBackbone(nn.Module):
    """Full-resolution Transformer encoder ablation.

    This keeps the same BPE embedding and MLM head used by ``DnaModel`` but
    removes the DualHelix microscope/telescope streams and bridge modules.
    """

    def __init__(self, config: DnaConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.head_dim = hidden_size // config.num_attention_heads
        self.rope_cache = RoPECache(self.head_dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerRoPE(
                    embed_dim=hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    dim_feedforward=4 * hidden_size,
                    dropout=config.dropout,
                    use_conv_ffn=config.use_conv_ffn,
                    conv_kernel_size=config.conv_kernel_size,
                    drop_path=config.drop_path_rate,
                    use_moe=config.use_moe,
                    moe_num_experts=config.moe_num_experts,
                    moe_top_k=config.moe_top_k,
                    moe_beta_z=config.moe_beta_z,
                )
                for _ in range(config.high_level_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        sin, cos = self.rope_cache(x.shape[1], x.device)
        moe_aux_losses: list[torch.Tensor] = []
        router_entropies: list[torch.Tensor] = []

        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                def layer_forward(hidden, sin_cache, cos_cache, mask, layer_module=layer):
                    result = layer_module(
                        hidden,
                        src_key_padding_mask=mask,
                        sin=sin_cache,
                        cos=cos_cache,
                    )
                    out = result[0]
                    aux = result[1] if result[1] is not None else out.new_zeros(())
                    ent = result[2] if result[2] is not None else out.new_zeros(())
                    has_aux = out.new_tensor(float(result[1] is not None))
                    has_ent = out.new_tensor(float(result[2] is not None))
                    return out, aux, ent, has_aux, has_ent

                x, aux, ent, has_aux, has_ent = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    x,
                    sin,
                    cos,
                    src_key_padding_mask,
                    use_reentrant=False,
                )
                if bool(has_aux.item()):
                    moe_aux_losses.append(aux)
                if bool(has_ent.item()):
                    router_entropies.append(ent)
            else:
                x, aux, ent = layer(
                    x,
                    src_key_padding_mask=src_key_padding_mask,
                    sin=sin,
                    cos=cos,
                )
                if aux is not None:
                    moe_aux_losses.append(aux)
                if ent is not None:
                    router_entropies.append(ent)

        out = self.final_norm(x)
        moe_aux_loss = torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        mean_router_entropy = torch.stack(router_entropies).mean() if router_entropies else None
        return out, moe_aux_loss, mean_router_entropy
