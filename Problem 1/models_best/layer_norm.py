"""
Efficient Layer Normalization
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias
    Slightly more efficient than nn.LayerNorm for inference
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = True):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
            bias: Whether to use bias parameter
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        
        if self.bias is not None:
            return self.weight * normalized + self.bias
        else:
            return self.weight * normalized


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    More efficient than LayerNorm, used in modern LLMs
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
