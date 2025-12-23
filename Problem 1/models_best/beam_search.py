"""
Beam Search Decoder for high-quality translation
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BeamSearchDecoder:
    """
    Beam Search decoder with length penalty
    
    Generates better translations than greedy search by exploring
    multiple hypotheses simultaneously
    """
    
    def __init__(self, model, beam_size: int = 5, max_len: int = 200,
                 length_penalty: float = 0.6, bos_idx: int = 1,
                 eos_idx: int = 2, pad_idx: int = 0):
        """
        Args:
            model: Transformer model
            beam_size: Number of beams to keep
            max_len: Maximum decoding length
            length_penalty: Length penalty factor (0.0 = no penalty, higher = prefer longer)
            bos_idx: Begin-of-sentence token index
            eos_idx: End-of-sentence token index
            pad_idx: Padding token index
        """
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
    
    def decode(self, src: torch.Tensor, src_mask: torch.Tensor) -> List[int]:
        """
        Decode a single source sequence using beam search
        
        Args:
            src: [1, src_len]
            src_mask: [1, 1, 1, src_len]
        
        Returns:
            best_sequence: List of token indices
        """
        device = src.device
        batch_size = 1  # Beam search for single sentence
        
        # Encode source
        with torch.no_grad():
            enc_output = self.model.encode(src, src_mask)
            # [1, src_len, d_model]
        
        # Initialize beams with BOS token
        beams = [(torch.tensor([[self.bos_idx]], device=device), 0.0)]
        # List of (sequence, score) tuples
        
        completed_beams = []
        
        for step in range(self.max_len):
            all_candidates = []
            
            for seq, score in beams:
                # If sequence already ended, add to completed
                if seq[0, -1].item() == self.eos_idx:
                    completed_beams.append((seq, score))
                    continue
                
                # Generate target mask
                tgt_len = seq.size(1)
                tgt_mask = self.model.generate_subsequent_mask(tgt_len).to(device)
                # [1, tgt_len, tgt_len]
                
                # Expand encoder output for beam
                # enc_output: [1, src_len, d_model]
                
                # Decode
                with torch.no_grad():
                    output = self.model.decoder(seq, enc_output, src_mask, tgt_mask)
                    # [1, tgt_len, vocab_size]
                
                # Get log probabilities for next token
                logits = output[0, -1, :]  # [vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k tokens
                topk_log_probs, topk_indices = torch.topk(log_probs, self.beam_size)
                
                # Create candidates
                for k in range(self.beam_size):
                    token_id = topk_indices[k].unsqueeze(0).unsqueeze(0)  # [1, 1]
                    new_seq = torch.cat([seq, token_id], dim=1)
                    new_score = score + topk_log_probs[k].item()
                    all_candidates.append((new_seq, new_score))
            
            # If no active beams, stop
            if not all_candidates:
                break
            
            # Select top beams
            # Apply length penalty: score / (length ** length_penalty)
            scored_candidates = []
            for seq, score in all_candidates:
                length = seq.size(1)
                normalized_score = score / (length ** self.length_penalty)
                scored_candidates.append((seq, score, normalized_score))
            
            # Sort by normalized score
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Keep top beam_size beams
            beams = [(seq, score) for seq, score, _ in scored_candidates[:self.beam_size]]
            
            # Check if all beams completed
            if len(beams) == 0:
                break
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Select best completed beam
        if completed_beams:
            best_beam = max(
                completed_beams,
                key=lambda x: x[1] / (x[0].size(1) ** self.length_penalty)
            )
            best_sequence = best_beam[0][0].tolist()
            
            # Remove BOS and EOS tokens
            if best_sequence[0] == self.bos_idx:
                best_sequence = best_sequence[1:]
            if best_sequence and best_sequence[-1] == self.eos_idx:
                best_sequence = best_sequence[:-1]
            
            return best_sequence
        else:
            # Fallback to empty sequence
            return []
    
    def decode_batch(self, src: torch.Tensor, src_mask: torch.Tensor) -> List[List[int]]:
        """
        Decode a batch of source sequences
        
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
        
        Returns:
            sequences: List of decoded sequences (list of token indices)
        """
        batch_size = src.size(0)
        results = []
        
        for i in range(batch_size):
            src_i = src[i:i+1]  # [1, src_len]
            src_mask_i = src_mask[i:i+1]  # [1, 1, 1, src_len]
            sequence = self.decode(src_i, src_mask_i)
            results.append(sequence)
        
        return results


class GreedyDecoder:
    """
    Greedy decoder (faster but lower quality than beam search)
    """
    
    def __init__(self, model, max_len: int = 200, bos_idx: int = 1,
                 eos_idx: int = 2):
        """
        Args:
            model: Transformer model
            max_len: Maximum decoding length
            bos_idx: Begin-of-sentence token index
            eos_idx: End-of-sentence token index
        """
        self.model = model
        self.max_len = max_len
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
    
    def decode(self, src: torch.Tensor, src_mask: torch.Tensor) -> List[int]:
        """
        Decode a single source sequence using greedy search
        
        Args:
            src: [1, src_len]
            src_mask: [1, 1, 1, src_len]
        
        Returns:
            sequence: List of token indices
        """
        device = src.device
        
        # Encode source
        with torch.no_grad():
            enc_output = self.model.encode(src, src_mask)
        
        # Initialize with BOS token
        tgt = torch.tensor([[self.bos_idx]], device=device)
        
        for _ in range(self.max_len):
            # Generate target mask
            tgt_len = tgt.size(1)
            tgt_mask = self.model.generate_subsequent_mask(tgt_len).to(device)
            
            # Decode
            with torch.no_grad():
                output = self.model.decoder(tgt, enc_output, src_mask, tgt_mask)
                # [1, tgt_len, vocab_size]
            
            # Get next token (greedy)
            next_token = output[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == self.eos_idx:
                break
        
        # Convert to list and remove BOS/EOS
        sequence = tgt[0].tolist()
        if sequence[0] == self.bos_idx:
            sequence = sequence[1:]
        if sequence and sequence[-1] == self.eos_idx:
            sequence = sequence[:-1]
        
        return sequence
    
    def decode_batch(self, src: torch.Tensor, src_mask: torch.Tensor) -> List[List[int]]:
        """
        Decode a batch of source sequences
        
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
        
        Returns:
            sequences: List of decoded sequences
        """
        batch_size = src.size(0)
        results = []
        
        for i in range(batch_size):
            src_i = src[i:i+1]
            src_mask_i = src_mask[i:i+1]
            sequence = self.decode(src_i, src_mask_i)
            results.append(sequence)
        
        return results
