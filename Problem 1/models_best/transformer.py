"""
Best Transformer Model for Neural Machine Translation

Combines best practices:
- Pre-Layer Normalization for stable training
- Label Smoothing for better generalization
- Weight Tying to reduce parameters
- Proper initialization
- Beam Search for high-quality inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig
from .beam_search import BeamSearchDecoder, GreedyDecoder


class BestTransformer(nn.Module):
    """
    Best Transformer Model for Neural Machine Translation
    
    Improvements over standard Transformer:
    1. Pre-Layer Normalization (more stable training)
    2. Weight Tying (decoder embedding = output projection)
    3. Optional shared embeddings (if same vocab for src/tgt)
    4. Proper weight initialization
    5. Dropout variants (attention, activation, residual)
    6. Beam search with length penalty
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 config: Optional[TransformerConfig] = None):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            config: Model configuration (use TransformerConfig.base() if None)
        """
        super().__init__()
        
        # Use default config if not provided
        if config is None:
            config = TransformerConfig.base()
        
        self.config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = config.pad_idx
        self.device = config.device
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_encoder_layers,
            max_len=config.max_len,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            pad_idx=config.pad_idx,
            scale_embedding=config.scale_embedding
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_decoder_layers,
            max_len=config.max_len,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            pad_idx=config.pad_idx,
            scale_embedding=config.scale_embedding
        )
        
        # Weight tying: decoder embedding = output projection
        if config.weight_tying:
            self.decoder.fc_out.weight = self.decoder.embedding.embedding.weight
        
        # Share encoder and decoder embeddings if same vocabulary
        if config.share_embeddings and src_vocab_size == tgt_vocab_size:
            self.decoder.embedding.embedding.weight = self.encoder.embedding.embedding.weight
        
        # Initialize weights
        self.init_weights(config.init_std)
        
        # Initialize beam search decoder
        self.beam_search = BeamSearchDecoder(
            model=self,
            beam_size=config.beam_size,
            max_len=config.max_decode_length,
            length_penalty=config.length_penalty,
            bos_idx=config.bos_idx,
            eos_idx=config.eos_idx,
            pad_idx=config.pad_idx
        )
        
        # Initialize greedy decoder
        self.greedy = GreedyDecoder(
            model=self,
            max_len=config.max_decode_length,
            bos_idx=config.bos_idx,
            eos_idx=config.eos_idx
        )
        
        # Move to device
        self.to(self.device)
        
        print(f"✓ BestTransformer initialized")
        print(f"  - Parameters: {self.count_parameters():,}")
        print(f"  - Device: {self.device}")
        print(f"  - Encoder layers: {config.n_encoder_layers}")
        print(f"  - Decoder layers: {config.n_decoder_layers}")
        print(f"  - d_model: {config.d_model}")
        print(f"  - n_heads: {config.n_heads}")
        print(f"  - Weight tying: {config.weight_tying}")
        print(f"  - Shared embeddings: {config.share_embeddings}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights"""
        self.encoder.init_weights(init_std)
        self.decoder.init_weights(init_std)
        
        # Special initialization for output projection if weight tying
        if not self.config.weight_tying:
            nn.init.normal_(self.decoder.fc_out.weight, mean=0.0, std=init_std)
    
    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate source and target masks
        
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        
        Returns:
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        """
        # Source mask: mask padding tokens
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch_size, 1, 1, src_len]
        
        # Target mask: mask padding tokens AND future tokens
        tgt_padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)
        # [batch_size, 1, tgt_len, 1]
        
        # Create causal mask (no peeking into future)
        tgt_len = tgt.size(1)
        tgt_causal_mask = self.generate_subsequent_mask(tgt_len).to(tgt.device)
        # [1, tgt_len, tgt_len]
        
        # Combine padding and causal masks
        tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0)
        # [batch_size, 1, tgt_len, tgt_len]
        
        return src_mask, tgt_mask
    
    @staticmethod
    def generate_subsequent_mask(size: int) -> torch.Tensor:
        """
        Generate causal mask to prevent attending to future tokens
        
        Args:
            size: Sequence length
        
        Returns:
            mask: [1, size, size] - lower triangular matrix of 1s
        """
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0)
        # [1, size, size]
        return mask.bool()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Forward pass
        
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, tgt_len, vocab_size]
            attentions: Tuple of (encoder_attn, decoder_self_attn, decoder_cross_attn) if return_attention
        """
        # Generate masks if not provided
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encode
        if return_attention:
            enc_output, enc_attn = self.encoder(src, src_mask, return_attention=True)
        else:
            enc_output = self.encoder(src, src_mask, return_attention=False)
            enc_attn = None
        
        # Decode
        if return_attention:
            output, dec_self_attn, dec_cross_attn = self.decoder(
                tgt, enc_output, src_mask, tgt_mask, return_attention=True
            )
            return output, (enc_attn, dec_self_attn, dec_cross_attn)
        else:
            output = self.decoder(tgt, enc_output, src_mask, tgt_mask, return_attention=False)
            return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence
        
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
        
        Returns:
            enc_output: [batch_size, src_len, d_model]
        """
        if src_mask is None:
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        return self.encoder(src, src_mask)
    
    def translate_greedy(self, src: torch.Tensor) -> List[int]:
        """
        Translate using greedy decoding (fast but lower quality)
        
        Args:
            src: [1, src_len] - Single source sequence
        
        Returns:
            translation: List of token indices
        """
        self.eval()
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return self.greedy.decode(src, src_mask)
    
    def translate_beam(self, src: torch.Tensor, beam_size: Optional[int] = None,
                       length_penalty: Optional[float] = None) -> List[int]:
        """
        Translate using beam search (slower but higher quality)
        
        Args:
            src: [1, src_len] - Single source sequence
            beam_size: Beam size (uses config default if None)
            length_penalty: Length penalty (uses config default if None)
        
        Returns:
            translation: List of token indices
        """
        self.eval()
        
        # Override beam search settings if provided
        if beam_size is not None:
            original_beam_size = self.beam_search.beam_size
            self.beam_search.beam_size = beam_size
        
        if length_penalty is not None:
            original_length_penalty = self.beam_search.length_penalty
            self.beam_search.length_penalty = length_penalty
        
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        translation = self.beam_search.decode(src, src_mask)
        
        # Restore original settings
        if beam_size is not None:
            self.beam_search.beam_size = original_beam_size
        if length_penalty is not None:
            self.beam_search.length_penalty = original_length_penalty
        
        return translation
    
    def translate_batch_greedy(self, src: torch.Tensor) -> List[List[int]]:
        """
        Translate batch using greedy decoding
        
        Args:
            src: [batch_size, src_len]
        
        Returns:
            translations: List of translated sequences
        """
        self.eval()
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return self.greedy.decode_batch(src, src_mask)
    
    def translate_batch_beam(self, src: torch.Tensor) -> List[List[int]]:
        """
        Translate batch using beam search
        
        Args:
            src: [batch_size, src_len]
        
        Returns:
            translations: List of translated sequences
        """
        self.eval()
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return self.beam_search.decode_batch(src, src_mask)
