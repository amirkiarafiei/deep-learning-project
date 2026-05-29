"""Mamba (selective state-space) bi-temporal fusion for Track 4 v6.

Given spatial feature maps from a Siamese ConvNeXt backbone for image A
and image B (each ``(B, C, H, W)``), tokenize and interleave them into a
single sequence ``[A_pos_0, B_pos_0, A_pos_1, B_pos_1, ...]``, then pass
through a stack of Mamba SSM blocks. This forces the SSM hidden state to
encode the A→B transition at every spatial position via selective
recurrence — a fundamentally different inductive bias from cross-attention
fusion (v5) or concat-difference (v1).

Reference impls:
- Mamba paper: Gu & Dao 2023, arXiv:2312.00752
- mamba-ssm pip package: state-spaces/mamba

If mamba-ssm cannot be imported (e.g. local CPU venv), this module falls
back to a 1-D depthwise-conv SSM-style block (no selective scan; slower
on long sequences but functionally close enough for smoke-testing).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore
    _MAMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for environments without CUDA kernels
    _MAMBA_AVAILABLE = False


class _MambaBlock(nn.Module):
    """LayerNorm pre-norm + Mamba SSM mixer + residual + (optional) MLP."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if _MAMBA_AVAILABLE:
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # Fallback: per-token affine + 1-D depthwise conv (no selective scan).
            self.mixer = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv - 1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mixer.
        if _MAMBA_AVAILABLE:
            mix = self.mixer(self.norm1(x))
        else:
            # Conv1d expects (B, C, L)
            h = self.norm1(x).transpose(1, 2)
            mix = self.mixer(h)[:, :, : x.size(1)].transpose(1, 2)
        x = x + self.drop(mix)
        # MLP.
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class InterleavedMambaFusion(nn.Module):
    """Tokenize + interleave bi-temporal features, then ``n_blocks`` Mamba blocks.

    Args:
        in_channels: backbone channel dim (768 for ConvNeXt-Tiny last stage).
        d_model: SSM embedding dim. If different from ``in_channels`` we
            project; if equal, identity.
        n_blocks: number of Mamba blocks stacked over the interleaved sequence.
        d_state, d_conv, expand: Mamba hyperparameters (paper defaults).
        dropout: dropout in MLPs.
    """

    def __init__(
        self,
        in_channels: int = 768,
        d_model: int = 768,
        n_blocks: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = (
            nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        )
        # 2 segment embeddings (A / B) added to each token.
        self.segment = nn.Embedding(2, d_model)
        self.blocks = nn.ModuleList(
            [
                _MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def _tokenize(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, H*W, d_model)`` after projection."""
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)
        return x

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Return ``(B, d_model)`` mean-pooled change embedding."""
        tok_a = self._tokenize(feat_a)
        tok_b = self._tokenize(feat_b)

        # Segment embeddings.
        seg_a = self.segment.weight[0]
        seg_b = self.segment.weight[1]
        tok_a = tok_a + seg_a
        tok_b = tok_b + seg_b

        # Interleave: [A_0, B_0, A_1, B_1, ..., A_{N-1}, B_{N-1}]
        b, n, d = tok_a.shape
        seq = torch.stack([tok_a, tok_b], dim=2).reshape(b, 2 * n, d)

        # Mamba stack.
        for block in self.blocks:
            seq = block(seq)
        seq = self.final_norm(seq)

        # Mean-pool over the 2N tokens.
        return seq.mean(dim=1)
