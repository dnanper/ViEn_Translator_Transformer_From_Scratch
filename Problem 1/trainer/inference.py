"""
Inference Script for Vietnamese-English Translation

Translate sentences or files using trained model
"""

import torch
from pathlib import Path
import sys
import os
from typing import List, Optional
import time
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models_best import BestTransformer, TransformerConfig
from utils.data_processing import DataProcessor
from config import Config


class Translator:
    """
    Translator for inference using trained SentencePiece tokenizer
    """
    
    def __init__(self, checkpoint_path: str, tokenizer_dir: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            tokenizer_dir: Directory containing tokenizer model
            device: Device to run on
        """
        print("Loading translator...")
        
        # Load checkpoint (weights_only=False for PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        self.config.device = device
        
        # Initialize data processor with tokenizer
        self.processor = DataProcessor(Config)
        self.processor.load_tokenizer(tokenizer_dir)
        
        # Create model
        vocab_size = self.processor.vocab_size
        self.model = BestTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            config=self.config
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Device: {device}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text to token IDs using SentencePiece
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.processor.encode_sentence(text, add_sos=True, add_eos=True)
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text using SentencePiece
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.processor.decode_sentence(token_ids, skip_special_tokens=True)
    
    def translate(self, text: str, beam_size: int = 5, length_penalty: float = 0.6,
                  use_greedy: bool = False) -> str:
        """
        Translate a single sentence
        
        Args:
            text: Input text
            beam_size: Beam size for beam search
            length_penalty: Length penalty factor
            use_greedy: Use greedy search instead of beam search
        
        Returns:
            Translated text
        """
        # Tokenize input
        src_ids = self.tokenize(text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.config.device)
        
        # Translate
        start_time = time.time()
        
        with torch.no_grad():
            if use_greedy:
                tgt_ids = self.model.translate_greedy(src_tensor)
            else:
                tgt_ids = self.model.translate_beam(
                    src_tensor, 
                    beam_size=beam_size,
                    length_penalty=length_penalty
                )
        
        translation_time = time.time() - start_time
        
        # Detokenize
        translation = self.detokenize(tgt_ids)
        
        print(f"Translation time: {translation_time:.3f}s")
        
        return translation
    
    def translate_batch_greedy(self, src_batch: torch.Tensor, max_len: int = 256) -> List[List[int]]:
        """
        Batch greedy decoding - processes entire batch in parallel
        
        Args:
            src_batch: Source batch tensor [batch_size, src_len]
            max_len: Maximum decode length
            
        Returns:
            List of decoded token ID sequences
        """
        batch_size = src_batch.size(0)
        device = src_batch.device
        
        # Create source mask
        src_mask = (src_batch != self.processor.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Encode source batch once
        memory = self.model.encoder(self.model.src_embed(src_batch), src_mask)
        
        # Start with BOS token for all sequences
        decoder_input = torch.full((batch_size, 1), self.config.bos_idx, 
                                   dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            # Create causal mask
            tgt_len = decoder_input.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).bool()
            
            # Decode
            tgt_embed = self.model.tgt_embed(decoder_input)
            output = self.model.decoder(tgt_embed, memory, src_mask, tgt_mask)
            logits = self.model.output_projection(output)
            
            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch, 1]
            
            # Mark finished sequences (encountered EOS)
            finished = finished | (next_token.squeeze(1) == self.config.eos_idx)
            
            # Append next token
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if all finished
            if finished.all():
                break
        
        # Convert to list of token IDs
        results = []
        for i in range(batch_size):
            token_ids = decoder_input[i].cpu().tolist()
            results.append(token_ids)
        
        return results
    
    def translate_file(self, input_file: str, output_file: str,
                       beam_size: int = 5, use_greedy: bool = False,
                       batch_size: int = 1, save_jsonl: bool = True):
        """
        Translate a file line by line with batching support
        
        Args:
            input_file: Input file path
            output_file: Output file path (will save both .vi and .jsonl)
            beam_size: Beam size
            use_greedy: Use greedy search
            batch_size: Number of sentences to process at once
            save_jsonl: Save results in JSONL format for incremental saving
        """
        print(f"\nTranslating {input_file}...")
        print(f"Output will be saved to {output_file}")
        print(f"Batch size: {batch_size}")
        
        # Read all lines
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        # Prepare output files
        output_path = Path(output_file)
        jsonl_path = output_path.parent / (output_path.stem + '.jsonl')
        
        # Open files for incremental saving
        f_out = open(output_file, 'w', encoding='utf-8')
        if save_jsonl:
            f_jsonl = open(jsonl_path, 'w', encoding='utf-8')
        
        total_lines = len(lines)
        translated_count = 0
        
        try:
            # Process in batches
            for batch_start in tqdm(range(0, total_lines, batch_size), desc="Translating"):
                batch_end = min(batch_start + batch_size, total_lines)
                batch_lines = lines[batch_start:batch_end]
                
                batch_src_tensors = []
                batch_original = []
                
                # Prepare batch
                for line in batch_lines:
                    line = line.strip()
                    batch_original.append(line)
                    
                    if not line:
                        continue
                    
                    # Tokenize
                    src_ids = self.tokenize(line)
                    src_tensor = torch.tensor(src_ids, dtype=torch.long)
                    batch_src_tensors.append(src_tensor)
                
                # Skip if all empty
                if not batch_src_tensors:
                    for _ in batch_lines:
                        f_out.write('\n')
                        if save_jsonl:
                            f_jsonl.write(json.dumps({
                                'id': translated_count,
                                'source': '',
                                'translation': ''
                            }, ensure_ascii=False) + '\n')
                        translated_count += 1
                    continue
                
                # Pad batch to same length
                max_len = max(len(t) for t in batch_src_tensors)
                padded_batch = []
                for t in batch_src_tensors:
                    if len(t) < max_len:
                        padding = torch.full((max_len - len(t),), self.processor.pad_idx, dtype=torch.long)
                        t = torch.cat([t, padding])
                    padded_batch.append(t)
                
                # Stack into batch tensor [batch_size, seq_len]
                batch_tensor = torch.stack(padded_batch).to(self.config.device)
                
                # Translate batch
                with torch.no_grad():
                    if use_greedy:
                        # TRUE BATCH GREEDY - process entire batch in parallel
                        batch_results = self.translate_batch_greedy(batch_tensor, max_len=256)
                        batch_translations = [self.detokenize(tgt_ids) for tgt_ids in batch_results]
                    else:
                        # Beam search - process one by one (batching beam search is complex)
                        batch_translations = []
                        for i in range(batch_tensor.size(0)):
                            tgt_ids = self.model.translate_beam(
                                batch_tensor[i:i+1],
                                beam_size=beam_size,
                                length_penalty=0.6
                            )
                            translation = self.detokenize(tgt_ids)
                            batch_translations.append(translation)
                
                # Save results immediately
                trans_idx = 0
                for i, original in enumerate(batch_original):
                    if not original:
                        translation = ''
                    else:
                        translation = batch_translations[trans_idx]
                        trans_idx += 1
                    
                    # Write to text file
                    f_out.write(translation + '\n')
                    f_out.flush()  # Flush to disk immediately
                    
                    # Write to JSONL
                    if save_jsonl:
                        f_jsonl.write(json.dumps({
                            'id': translated_count,
                            'source': original,
                            'translation': translation
                        }, ensure_ascii=False) + '\n')
                        f_jsonl.flush()  # Flush to disk immediately
                    
                    # Print sample
                    if translated_count < 5 or (translated_count + 1) % 100 == 0:
                        print(f"\n[{translated_count + 1}] EN: {original[:100]}...")
                        print(f"    VI: {translation[:100]}...")
                    
                    translated_count += 1
        
        finally:
            # Close files
            f_out.close()
            if save_jsonl:
                f_jsonl.close()
        
        print(f"\n✓ Translated {translated_count} lines")
        print(f"✓ Translations saved to:")
        print(f"  - Text: {output_file}")
        if save_jsonl:
            print(f"  - JSONL: {jsonl_path}")


def main():
    """Interactive translation"""
    
    # Get project root directory (parent of trainer/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Configuration (relative to project root)
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_en2vi" / "best_model.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create translator
    translator = Translator(str(CHECKPOINT_PATH), str(TOKENIZER_DIR), DEVICE)
    
    print("\n" + "=" * 60)
    print("English-Vietnamese Translator (EN → VI)")
    print("=" * 60)
    print("Commands:")
    print("  - Type a sentence to translate")
    print("  - Type 'file <input> <output>' to translate a file")
    print("  - Type 'beam <size>' to change beam size (default: 5)")
    print("  - Type 'greedy' to toggle greedy search")
    print("  - Type 'exit' to quit")
    print("=" * 60)
    
    beam_size = 5
    use_greedy = False
    
    while True:
        try:
            user_input = input("\nInput: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('file '):
                # Translate file
                parts = user_input.split()
                if len(parts) != 3:
                    print("Usage: file <input_file> <output_file>")
                    continue
                
                input_file = parts[1]
                output_file = parts[2]
                translator.translate_file(input_file, output_file, beam_size, use_greedy)
            
            elif user_input.lower().startswith('beam '):
                # Change beam size
                try:
                    beam_size = int(user_input.split()[1])
                    print(f"Beam size set to {beam_size}")
                except:
                    print("Invalid beam size")
            
            elif user_input.lower() == 'greedy':
                # Toggle greedy search
                use_greedy = not use_greedy
                print(f"Greedy search: {'ON' if use_greedy else 'OFF'}")
            
            else:
                # Translate sentence
                print(f"\nTranslating (beam_size={beam_size}, greedy={use_greedy})...")
                translation = translator.translate(user_input, beam_size, use_greedy=use_greedy)
                print(f"\nTranslation: {translation}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
