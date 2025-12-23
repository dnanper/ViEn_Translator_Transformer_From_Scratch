"""
Transformer Decoder with Pre-Layer Normalization
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .embeddings import ScaledEmbedding
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Pre-Layer Normalization
    
    Structure:
    1. LayerNorm -> Masked Self-Attention -> Dropout -> Residual
    2. LayerNorm -> Cross-Attention -> Dropout -> Residual
    3. LayerNorm -> Feed-Forward -> Dropout -> Residual
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
        self.norm3 = LayerNorm(d_model)
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, attention_dropout
        )
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout, attention_dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout, activation_dropout)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            x: [batch_size, tgt_len, d_model]
            enc_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, tgt_len, d_model]
            self_attn: [batch_size, n_heads, tgt_len, tgt_len] (if return_attention)
            cross_attn: [batch_size, n_heads, tgt_len, src_len] (if return_attention)
        """
        # Masked self-attention block with pre-norm
        residual = x
        x = self.norm1(x)
        
        if return_attention:
            self_attn_out, self_attn = self.self_attn(
                x, x, x, tgt_mask, return_attention=True
            )
            x = residual + self.dropout1(self_attn_out)
        else:
            x = residual + self.dropout1(self.self_attn(x, x, x, tgt_mask))
            self_attn = None
        
        # Cross-attention block with pre-norm
        residual = x
        x = self.norm2(x)
        
        if return_attention:
            cross_attn_out, cross_attn = self.cross_attn(
                x, enc_output, enc_output, src_mask, return_attention=True
            )
            x = residual + self.dropout2(cross_attn_out)
        else:
            x = residual + self.dropout2(
                self.cross_attn(x, enc_output, enc_output, src_mask)
            )
            cross_attn = None
        
        # Feed-forward block with pre-norm
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout3(self.ffn(x))
        
        if return_attention:
            return x, self_attn, cross_attn
        return x


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, max_len: int = 5000,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.0, pad_idx: int = 0,
                 scale_embedding: bool = True):
        """
        Args:
            vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            n_layers: Number of decoder layers
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
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, dropout,
                attention_dropout, activation_dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (important for Pre-LN)
        self.final_norm = LayerNorm(d_model)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            tgt: [batch_size, tgt_len]
            enc_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, tgt_len, vocab_size]
            self_attentions: List of self-attention weights (if return_attention)
            cross_attentions: List of cross-attention weights (if return_attention)
        """
        # Embedding + positional encoding
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        self_attentions = [] if return_attention else None
        cross_attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, self_attn, cross_attn = layer(
                    x, enc_output, src_mask, tgt_mask, return_attention=True
                )
                self_attentions.append(self_attn)
                cross_attentions.append(cross_attn)
            else:
                x = layer(x, enc_output, src_mask, tgt_mask, return_attention=False)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        output = self.fc_out(x)
        
        if return_attention:
            return output, self_attentions, cross_attentions
        return output
    
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
