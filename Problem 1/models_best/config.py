"""
Configuration for Best Transformer Model
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""
    
    # Model architecture
    d_model: int = 512
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    
    # Position encoding
    max_len: int = 5000
    
    # Training
    label_smoothing: float = 0.1
    weight_tying: bool = True
    share_embeddings: bool = False  # Share src and tgt embeddings (if same vocab)
    
    # Initialization
    init_std: float = 0.02
    scale_embedding: bool = True
    
    # Optimization
    warmup_steps: int = 4000
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Inference
    beam_size: int = 5
    length_penalty: float = 0.6
    max_decode_length: int = 200
    
    # Device
    device: str = 'cuda'
    pad_idx: int = 0
    bos_idx: int = 2  # SOS token
    eos_idx: int = 3  # EOS token
    
    @classmethod
    def small(cls) -> 'TransformerConfig':
        """Small model configuration (fast, ~60M params)"""
        return cls(
            d_model=256,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.3,
            attention_dropout=0.1,
            activation_dropout=0.1,
            warmup_steps=4000,
        )
    
    @classmethod
    def base(cls) -> 'TransformerConfig':
        """Base model configuration (balanced, ~65M params)"""
        return cls(
            d_model=512,
            n_encoder_layers=6,
            n_decoder_layers=6,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            warmup_steps=8000,
        )
    
    @classmethod
    def large(cls) -> 'TransformerConfig':
        """Large model configuration (best quality, ~213M params)"""
        return cls(
            d_model=1024,
            n_encoder_layers=6,
            n_decoder_layers=6,
            n_heads=16,
            d_ff=4096,
            dropout=0.3,
            attention_dropout=0.1,
            activation_dropout=0.1,
            warmup_steps=16000,
            learning_rate=5e-4,
        )
    
    @classmethod
    def deep(cls) -> 'TransformerConfig':
        """Deep model configuration (12 layers)"""
        return cls(
            d_model=512,
            n_encoder_layers=12,
            n_decoder_layers=12,
            n_heads=8,
            d_ff=2048,
            dropout=0.2,
            attention_dropout=0.1,
            activation_dropout=0.1,
            warmup_steps=16000,
        )
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
        assert 0 <= self.label_smoothing < 1, "label_smoothing must be in [0, 1)"
