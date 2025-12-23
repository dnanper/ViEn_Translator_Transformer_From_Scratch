"""
Multi-Head Attention with Improvements
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Improved Multi-Head Attention
    - Separate dropout for attention weights
    - Option to return attention weights for visualization
    - Efficient implementation
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model] (seq_len_k == seq_len_v)
            mask: [batch_size, 1, seq_len_q, seq_len_k] or broadcastable
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k] (if return_attention)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, H, L_q, d_k] @ [B, H, d_k, L_k] -> [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask (handle both Boolean and Float masks)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [B, H, L_q, L_k] @ [B, H, L_k, d_k] -> [B, H, L_q, d_k]
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        # [B, H, L_q, d_k] -> [B, L_q, H, d_k] -> [B, L_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    Uses single key and value projection shared across all heads
    Faster inference with minimal quality loss
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            dropout: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Query has n_heads, but key and value have single head
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, self.d_k, bias=True)
        self.W_v = nn.Linear(d_model, self.d_k, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k] or broadcastable
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k] (if return_attention)
        """
        batch_size = query.size(0)
        
        # Project query to multiple heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [B, H, L_q, d_k]
        
        # Project key and value to single head then expand
        K = self.W_k(key).unsqueeze(1)  # [B, 1, L_k, d_k]
        V = self.W_v(value).unsqueeze(1)  # [B, 1, L_v, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, H, L_q, L_k]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        # [B, H, L_q, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output
