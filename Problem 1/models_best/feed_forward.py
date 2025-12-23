"""
Position-wise Feed-Forward Network with Variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation_dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate for output
            activation_dropout: Dropout rate after activation
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation_dropout = nn.Dropout(activation_dropout) if activation_dropout > 0 else None
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = F.relu(self.fc1(x))
        
        if self.activation_dropout is not None:
            x = self.activation_dropout(x)
        
        x = self.fc2(x)
        x = self.output_dropout(x)
        
        return x


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit Feed-Forward Network
    Better than standard FFN in some cases
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation_dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate for output
            activation_dropout: Dropout rate after activation
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_ff)
        self.fc3 = nn.Linear(d_ff, d_model)
        self.activation_dropout = nn.Dropout(activation_dropout) if activation_dropout > 0 else None
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # GLU: gate = sigmoid(W_1 * x), value = W_2 * x
        # output = W_3 * (gate * value)
        gate = torch.sigmoid(self.fc1(x))
        value = self.fc2(x)
        x = gate * value
        
        if self.activation_dropout is not None:
            x = self.activation_dropout(x)
        
        x = self.fc3(x)
        x = self.output_dropout(x)
        
        return x


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network
    Used in PaLM, LLaMA - shows better performance than ReLU
    SwiGLU(x) = Swish(xW) ⊗ (xV) where Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation_dropout: float = 0.0, bias: bool = True):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (will be adjusted to make it divisible by 2/3 scaling)
            dropout: Dropout rate for output
            activation_dropout: Dropout rate after activation
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        # Following LLaMA, use 2/3 * 4 * d_model for d_ff
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, d_model, bias=bias)
        
        self.activation_dropout = nn.Dropout(activation_dropout) if activation_dropout > 0 else None
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # SwiGLU: swish(xW) * (xV)
        swish = F.silu(self.fc1(x))  # SiLU is same as Swish
        value = self.fc2(x)
        x = swish * value
        
        if self.activation_dropout is not None:
            x = self.activation_dropout(x)
        
        x = self.fc3(x)
        x = self.output_dropout(x)
        
        return x


class GELUFeedForward(nn.Module):
    """
    Feed-Forward with GELU activation
    Used in BERT, GPT - smoother than ReLU
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation_dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate for output
            activation_dropout: Dropout rate after activation
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation_dropout = nn.Dropout(activation_dropout) if activation_dropout > 0 else None
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = F.gelu(self.fc1(x))
        
        if self.activation_dropout is not None:
            x = self.activation_dropout(x)
        
        x = self.fc2(x)
        x = self.output_dropout(x)
        
        return x
