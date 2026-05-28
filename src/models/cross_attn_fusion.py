"""Cross-attention bi-temporal fusion for Track 3 v5.

Given patch-token sequences from a Siamese backbone for image A and image B,
produces a unified token sequence that downstream ML-Decoder heads attend to.

Design:
1. Block 1: Q from A, K/V from B  ->  A_cond ("A seen through B's lens")
2. Block 2: Q from B, K/V from A  ->  B_cond
3. Diff token: pool(A_cond - B_cond) -> 1 token
4. Concat [A_cond; B_cond; Diff] -> 2N+1 tokens
5. 2 transformer encoder layers integrate the joint sequence

The output token sequence is consumed by per-family ML-Decoder heads. The
sequence also has a pooled summary used by the binary changeflag head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    """LayerNorm pre-norm cross-attention with residual + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        x = q + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


class CrossAttentionFusion(nn.Module):
    """Two-direction cross-attention bi-temporal fusion + integration encoder.

    Args:
        d_model: token embedding dim (matches backbone output, e.g. 768 for DINOv2-Base).
        n_heads: number of attention heads in cross-attn + encoder.
        n_encoder_layers: number of self-attention encoder layers after cross-attn.
        dropout: dropout in cross-attn / FFN / encoder.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_encoder_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Two directional cross-attention blocks.
        self.cross_ab = _CrossAttnBlock(d_model, n_heads, dropout=dropout)
        self.cross_ba = _CrossAttnBlock(d_model, n_heads, dropout=dropout)

        # Difference-token projection (project pooled (A_cond - B_cond) -> 1 token).
        self.diff_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # 3 segment embeddings: A_cond / B_cond / diff. Helps the encoder
        # distinguish source.
        self.segment = nn.Embedding(3, d_model)

        # Self-attention integration encoder over [A_cond; B_cond; diff].
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        """Return the joint token sequence, shape ``(B, 2N+1, d_model)``."""
        # Cross-attention in both directions.
        a_cond = self.cross_ab(tokens_a, tokens_b)
        b_cond = self.cross_ba(tokens_b, tokens_a)

        # Diff token: mean-pool the difference, then project.
        diff_pooled = (a_cond - b_cond).mean(dim=1, keepdim=True)  # (B, 1, d)
        diff_tok = self.diff_proj(diff_pooled)

        # Add segment embeddings.
        seg_a = self.segment.weight[0]  # (d,)
        seg_b = self.segment.weight[1]
        seg_d = self.segment.weight[2]
        a_cond = a_cond + seg_a
        b_cond = b_cond + seg_b
        diff_tok = diff_tok + seg_d

        # Concat and integrate.
        seq = torch.cat([a_cond, b_cond, diff_tok], dim=1)  # (B, 2N+1, d)
        seq = self.encoder(seq)
        seq = self.final_norm(seq)
        return seq
