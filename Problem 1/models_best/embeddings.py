"""
Embeddings with Scaling and Dropout
"""

import torch
import torch.nn as nn
import math


class ScaledEmbedding(nn.Module):
    """
    Embedding layer with optional scaling
    """
    
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0, 
                 scale: bool = True, dropout: float = 0.0):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            pad_idx: Padding token index
            scale: Whether to scale embeddings by sqrt(d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model
        self.scale_factor = math.sqrt(d_model) if scale else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            embedded: [batch_size, seq_len, d_model]
        """
        embedded = self.embedding(x) * self.scale_factor
        
        if self.dropout is not None:
            embedded = self.dropout(embedded)
        
        return embedded
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize embedding weights"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        # Zero out padding embedding
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
