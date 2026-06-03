import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PEG(nn.Module):
    """Prompt Embedding Generator: Transformer decoder over image tokens."""

    def __init__(self, d_model: int = 256, num_queries: int = 1, num_layers: int = 5, dropout: float = 0.15, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Query content embeddings with orthogonal initialization
        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        if num_queries > 1:
            # Orthogonal initialization for better query diversity
            nn.init.orthogonal_(self.query_embed)
        else:
            # Fallback to kaiming for single query
            nn.init.kaiming_uniform_(self.query_embed, a=math.sqrt(5))

        # Learnable positional encoding for each query
        self.query_pos_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.query_pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.label_head = nn.Linear(d_model, 1)

    def forward(self, image_embed: torch.Tensor):
        """
        Args:
            image_embed: [B, HW, C] fused image embeddings.
        Returns:
            prompt_embed: [B, num_queries, d_model]
            labels: [B, num_queries] confidence scores.
        """
        B, _, _ = image_embed.shape
        # Combine content and positional embeddings for diverse queries
        # query_embed = (self.query_embed + self.query_pos_embed).unsqueeze(0).expand(B, -1, -1)
        query_embed = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        prompt_embed = self.transformer_decoder(query_embed, image_embed)
        labels = self.label_head(prompt_embed).squeeze(-1)
        return prompt_embed, labels

