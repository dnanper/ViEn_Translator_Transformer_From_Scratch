"""
Transformer Encoder with Pre-Layer Normalization
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .embeddings import ScaledEmbedding
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Pre-Layer Normalization
    
    Structure:
    1. LayerNorm -> Self-Attention -> Dropout -> Residual
    2. LayerNorm -> Feed-Forward -> Dropout -> Residual
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention
            activation_dropout: Dropout rate for FFN activation
        """
        super().__init__()
        
        # Pre-layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, attention_dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout, activation_dropout)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, 1, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, n_heads, seq_len, seq_len] (if return_attention)
        """
        # Self-attention block with pre-norm
        residual = x
        x = self.norm1(x)
        
        if return_attention:
            attn_out, attn_weights = self.self_attn(x, x, x, mask, return_attention=True)
            x = residual + self.dropout1(attn_out)
        else:
            x = residual + self.dropout1(self.self_attn(x, x, x, mask))
            attn_weights = None
        
        # Feed-forward block with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        
        if return_attention:
            return x, attn_weights
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, max_len: int = 5000,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.0, pad_idx: int = 0,
                 scale_embedding: bool = True):
        """
        Args:
            vocab_size: Source vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            n_layers: Number of encoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention
            activation_dropout: Dropout rate for FFN activation
            pad_idx: Padding token index
            scale_embedding: Whether to scale embeddings
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = ScaledEmbedding(
            vocab_size, d_model, pad_idx, scale=scale_embedding, dropout=0.0
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff, dropout,
                attention_dropout, activation_dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (important for Pre-LN)
        self.final_norm = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, src_len, d_model]
            attentions: List of attention weights (if return_attention)
        """
        # Embedding + positional encoding
        x = self.embedding(src)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, src_mask, return_attention=True)
                attentions.append(attn)
            else:
                x = layer(x, src_mask, return_attention=False)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        if return_attention:
            return x, attentions
        return x
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights"""
        # Initialize embeddings
        self.embedding.init_weights(init_std)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
