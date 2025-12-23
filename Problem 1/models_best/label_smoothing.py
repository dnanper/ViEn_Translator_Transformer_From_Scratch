"""
Label Smoothing Loss for better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    
    Instead of one-hot targets, use smoothed distribution:
    - Correct class: 1 - smoothing
    - Other classes: smoothing / (num_classes - 1)
    
    Reduces overconfidence and improves generalization
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1,
                 ignore_index: int = -100, reduction: str = 'mean'):
        """
        Args:
            num_classes: Number of classes (vocabulary size)
            smoothing: Smoothing factor (0.0 = no smoothing, 0.1 is common)
            ignore_index: Index to ignore (padding)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Confidence for correct class
        self.confidence = 1.0 - smoothing
        
        # Smoothing value for other classes
        self.smoothing_value = smoothing / (num_classes - 1)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size * seq_len, num_classes] or [batch_size, seq_len, num_classes]
            targets: [batch_size * seq_len] or [batch_size, seq_len]
        
        Returns:
            loss: scalar if reduction='mean' or 'sum', tensor if reduction='none'
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.size()
            logits = logits.contiguous().view(-1, num_classes)
            targets = targets.contiguous().view(-1)
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create mask for valid tokens (not padding)
        if self.ignore_index >= 0:
            mask = targets.ne(self.ignore_index)
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
        
        # Create smoothed target distribution
        with torch.no_grad():
            # CORRECT IMPLEMENTATION: Only operate on non-PAD positions
            smooth_targets = torch.zeros_like(log_probs)
            
            # Create mask for valid (non-padding) tokens
            non_pad = targets != self.ignore_index
            
            # Distribute smoothing over ALL classes (only for non-PAD rows)
            smooth_targets[non_pad] = self.smoothing / (self.num_classes - 1)
            
            # Set correct class to confidence (only for non-PAD rows)
            # This NEVER touches PAD rows - they remain all zeros
            smooth_targets[non_pad].scatter_(
                1,
                targets[non_pad].unsqueeze(1),
                self.confidence
            )
        
        # Compute KL divergence
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Apply mask for padding
        if self.ignore_index >= 0:
            mask = targets.ne(self.ignore_index)
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class CrossEntropyLoss(nn.Module):
    """
    Wrapper for standard CrossEntropyLoss with convenient interface
    """
    
    def __init__(self, ignore_index: int = -100, reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        Args:
            ignore_index: Index to ignore (padding)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (PyTorch 1.10+)
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size * seq_len, num_classes] or [batch_size, seq_len, num_classes]
            targets: [batch_size * seq_len] or [batch_size, seq_len]
        
        Returns:
            loss: scalar if reduction='mean' or 'sum', tensor if reduction='none'
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.size()
            logits = logits.contiguous().view(-1, num_classes)
            targets = targets.contiguous().view(-1)
        
        return self.loss_fn(logits, targets)
