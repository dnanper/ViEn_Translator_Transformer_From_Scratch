"""
Best Transformer Implementation for Neural Machine Translation
Combines best practices from research and industry
"""

from .transformer import BestTransformer
from .config import TransformerConfig
from .beam_search import BeamSearchDecoder
from .label_smoothing import LabelSmoothingLoss

__all__ = [
    'BestTransformer',
    'TransformerConfig',
    'BeamSearchDecoder',
    'LabelSmoothingLoss',
]

__version__ = '1.0.0'
