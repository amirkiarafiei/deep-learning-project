"""ML-Decoder multi-label classification head.

Reference:
    Ridnik, T., Sharir, G., Ben-Cohen, A., Ben-Baruch, E., & Noy, A. (2023).
    ML-Decoder: Scalable and Versatile Classification Head. WACV.

Each class is a learnable query embedding. The query attends to feature tokens
via cross-attention, then a per-class linear projection emits the logit.
Implicit label co-occurrence is modeled via self-attention between the
class queries.

For our small per-family class counts (12 / 12 / 24) we use a single
non-grouped decoder layer with a shared output projection — equivalent to
ML-Decoder with ``groups = 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLDecoder(nn.Module):
    """Transformer-decoder-based multi-label classification head.

    Args:
        num_classes: number of output classes (per family, e.g. 12 / 12 / 24).
        d_model: token embedding dim (matches fusion output).
        n_heads: number of attention heads in the decoder layer.
        n_layers: number of decoder layers stacked (default 1, the original paper).
        dropout: dropout in attention / FFN.
        zsl_init: if True, initialize class queries with Xavier; else with small
            normal noise. (Default Xavier.)
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # One learnable embedding per class.
        self.class_queries = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.xavier_uniform_(self.class_queries)

        # Transformer decoder: self-attn between queries + cross-attn to features.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output projection: one linear per query, shared weights across queries
        # would over-couple classes. We use an Linear(d_model, 1) applied to
        # every query position independently (achieved via simple matmul).
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-class logits, shape ``(B, num_classes)``.

        Args:
            features: (B, T, d_model) feature token sequence from the fusion module.
        """
        b = features.size(0)
        # Broadcast queries to the batch.
        queries = self.class_queries.unsqueeze(0).expand(b, -1, -1)  # (B, C, d)
        # Decode: tgt=queries, memory=features.
        out = self.decoder(tgt=queries, memory=features)  # (B, C, d)
        out = self.norm(out)
        logits = self.out_proj(out).squeeze(-1)  # (B, C)
        return logits
