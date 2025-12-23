"""
Train SentencePiece Tokenizer on PhoMT Dataset
Uses custom BPE + Unigram implementation from scratch
"""

import os
import re
import unicodedata
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import collections
import pickle

from byte_pair_encoder import BytePairEncoder
from sentence_piece import SentencePieceTrainer


class PhomtTokenizerTrainer:
    """Train tokenizer on PhoMT Vietnamese-English dataset"""
    
    def __init__(self, 
                 vocab_size=32000,
                 num_bpe_merges=10000,
                 max_samples=500000,
                 output_dir='./tokenizer_models'):
        """
        Args:
            vocab_size: Target vocabulary size for final tokenizer
            num_bpe_merges: Number of BPE merge operations
            max_samples: Maximum number of samples to use (None for all)
            output_dir: Directory to save trained tokenizer
        """
        self.vocab_size = vocab_size
        self.num_bpe_merges = num_bpe_merges
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bpe_encoder = BytePairEncoder()
        self.sp_trainer = SentencePieceTrainer()
        
    def normalize_text(self, text):
        """Normalize text (same as in data_processing.py)"""
        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def filter_pair(self, src, tgt, max_len=100):
        """Filter out bad sentence pairs"""
        src_len = len(src.split())
        tgt_len = len(tgt.split())
        
        # Remove empty or too long sentences
        if src_len == 0 or tgt_len == 0:
            return False
        if src_len > max_len or tgt_len > max_len:
            return False
        
        # Length ratio filtering
        length_ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
        if length_ratio > 2.0:
            return False
        
        return True
    
    def load_phomt_data(self):
        """
        Load and preprocess PhoMT dataset
        
        Returns:
            Combined text string from both English and Vietnamese
            (with proper sentence boundaries preserved)
        """
        print("=" * 80)
        print("LOADING PHOMT DATASET")
        print("=" * 80)
        
        # Load dataset from HuggingFace
        print("\n[1/3] Downloading dataset from HuggingFace (ura-hcmut/PhoMT)...")
        dataset = load_dataset("ura-hcmut/PhoMT")
        
        print(f"\n[2/3] Processing and filtering data...")
        print(f"   - Max samples: {self.max_samples if self.max_samples else 'ALL'}")
        print(f"   - Strategy: Shared vocabulary (both languages)")
        
        all_sentences = []
        sample_count = 0
        
        # Use train split only for tokenizer training
        for example in tqdm(dataset['train'], desc="Processing"):
            if self.max_samples and sample_count >= self.max_samples:
                break
                
            # Get English and Vietnamese text
            en_text = self.normalize_text(example['en'])
            vi_text = self.normalize_text(example['vi'])
            
            # Filter bad pairs
            if self.filter_pair(en_text, vi_text):
                # Add each sentence separately (preserving sentence boundaries)
                # This is correct for tokenizer training - each sentence is a unit
                all_sentences.append(en_text)
                all_sentences.append(vi_text)
                sample_count += 1
        
        print(f"\n[3/3] Loaded {sample_count:,} parallel sentence pairs")
        print(f"   - Total sentences for training: {len(all_sentences):,}")
        print(f"   - English sentences: {sample_count:,}")
        print(f"   - Vietnamese sentences: {sample_count:,}")
        
        # Combine all text with newlines to preserve sentence boundaries
        # Note: BPE will process this as continuous text, but sentence structure is maintained
        combined_text = ' '.join(all_sentences)
        
        print(f"\n✓ Total characters: {len(combined_text):,}")
        print(f"✓ Total words: {len(combined_text.split()):,}")
        print(f"✓ Average sentence length: {len(combined_text.split()) / len(all_sentences):.1f} words")
        
        return combined_text
    
    def train_bpe(self, text):
        """
        Train Byte Pair Encoder (Step 1)
        
        Args:
            text: Training text
            
        Returns:
            tokens: Counter of tokens
            characters: Set of base characters
        """
        print("\n" + "=" * 80)
        print("STEP 1: BYTE PAIR ENCODING (BPE)")
        print("=" * 80)
        
        print(f"\n[BPE] Training with {self.num_bpe_merges:,} merge operations...")
        
        # Train BPE
        self.bpe_encoder.fit(text, self.num_bpe_merges)
        
        tokens = self.bpe_encoder.get_tokens
        characters = self.bpe_encoder.get_characters
        
        print(f"\n✓ BPE Training Complete!")
        print(f"   - Vocabulary size: {len(tokens):,}")
        print(f"   - Base characters: {len(characters)}")
        print(f"   - Merge operations: {self.num_bpe_merges:,}")
        
        # Show some examples
        print("\n📝 Sample tokens (sorted by frequency):")
        most_common = tokens.most_common(20)
        for i, (token, count) in enumerate(most_common[:10], 1):
            token_display = token.replace('_', '▁').replace('\n', '\\n')
            print(f"   {i:2d}. '{token_display}' → {count:,} occurrences")
        
        return tokens, characters
    
    def train_unigram(self, text, tokens, characters):
        """
        Optimize tokenization with EM Algorithm (Step 2)
        
        Note: This uses EM to optimize probabilities and find best tokenization,
        NOT to generate new tokens (tokens come from BPE in Step 1)
        
        Args:
            text: Training text
            tokens: Counter of tokens from BPE
            characters: Set of base characters
        """
        print("\n" + "=" * 80)
        print("STEP 2: EM ALGORITHM (TOKENIZATION OPTIMIZATION)")
        print("=" * 80)
        
        print(f"\n[EM] Refining vocabulary with EM algorithm...")
        print(f"   - Initial vocab size: {len(tokens):,}")
        print(f"   - Target vocab size: {self.vocab_size:,}")
        print(f"   - Pruning strategy: Iterative EM + Pruning rounds")
        print(f"   - Note: Tokens from BPE, EM optimizes probabilities")
        
        # Train SentencePiece with Unigram + EM
        self.sp_trainer.fit(
            text=text,
            tokens=tokens,
            characters=characters,
            vocab_size=self.vocab_size,
            delta=0.01,      # Convergence threshold
            max_iter=10,     # Max EM iterations per round
            max_rounds=5     # Max pruning rounds
        )
        
        print(f"\n✓ Unigram Training Complete!")
        print(f"   - Final vocabulary size: {self.sp_trainer.vocab_size:,}")
        print(f"   - Maximum token length: {self.sp_trainer.maxlen}")
    
    def save_tokenizer(self, tokens, characters):
        """
        Save trained tokenizer to disk
        
        Args:
            tokens: Final token counter
            characters: Base character set
        """
        print("\n" + "=" * 80)
        print("SAVING TOKENIZER")
        print("=" * 80)
        
        # Save BPE encoder
        bpe_path = self.output_dir / 'bpe_encoder.pkl'
        with open(bpe_path, 'wb') as f:
            pickle.dump(self.bpe_encoder, f)
        print(f"\n✓ Saved BPE encoder: {bpe_path}")
        
        # Save SentencePiece trainer
        sp_path = self.output_dir / 'sentencepiece_trainer.pkl'
        with open(sp_path, 'wb') as f:
            pickle.dump(self.sp_trainer, f)
        print(f"✓ Saved SentencePiece trainer: {sp_path}")
        
        # Save vocabulary as text file for inspection
        vocab_path = self.output_dir / 'vocabulary.txt'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write(f"# Vocabulary Size: {len(tokens)}\n")
            f.write(f"# Training Date: {self._get_timestamp()}\n")
            f.write(f"# BPE Merges: {self.num_bpe_merges}\n")
            f.write(f"# Target Vocab Size: {self.vocab_size}\n")
            f.write("\n")
            
            for token, count in tokens.most_common():
                token_display = token.replace('_', '▁')
                f.write(f"{token_display}\t{count}\n")
        
        print(f"✓ Saved vocabulary: {vocab_path}")
        
        # Save metadata
        metadata = {
            'vocab_size': self.sp_trainer.vocab_size,
            'num_bpe_merges': self.num_bpe_merges,
            'max_token_length': self.sp_trainer.maxlen,
            'num_characters': len(characters),
            'training_samples': self.max_samples,
            'timestamp': self._get_timestamp()
        }
        
        metadata_path = self.output_dir / 'metadata.txt'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Saved metadata: {metadata_path}")
        
        print(f"\n📦 All files saved to: {self.output_dir.absolute()}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def test_tokenizer(self, test_sentences):
        """
        Test the trained tokenizer on sample sentences
        
        Args:
            test_sentences: List of test sentences
        """
        print("\n" + "=" * 80)
        print("TESTING TOKENIZER")
        print("=" * 80)
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n[Test {i}] Input: {sentence}")
            
            try:
                tokens = self.sp_trainer.tokenize(sentence, nbest_size=1)
                tokens_display = [t.replace('_', '▁') for t in tokens]
                print(f"   → Tokens ({len(tokens)}): {tokens_display}")
            except Exception as e:
                print(f"   ✗ Error: {e}")
    
    def train(self):
        """
        Main training pipeline
        
        Steps:
            1. Load PhoMT dataset (shared vocabulary approach)
            2. Train BPE encoder (generate initial vocabulary)
            3. Optimize with EM algorithm (refine tokenization)
            4. Save tokenizer
            5. Test tokenizer
        """
        print("\n" + "=" * 80)
        print("🚀 PHOMT TOKENIZER TRAINING PIPELINE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"   - Vocabulary Size: {self.vocab_size:,}")
        print(f"   - BPE Merges: {self.num_bpe_merges:,}")
        print(f"   - Max Samples: {self.max_samples if self.max_samples else 'ALL'}")
        print(f"   - Output Directory: {self.output_dir.absolute()}")
        print(f"   - Approach: Shared vocabulary (En + Vi)")
        print(f"\n💡 Training Strategy:")
        print(f"   Each sentence (En or Vi) is treated as a separate unit.")
        print(f"   Tokenizer learns patterns from both languages simultaneously.")
        
        # Step 1: Load data
        text = self.load_phomt_data()
        
        # Step 2: Train BPE
        tokens, characters = self.train_bpe(text)
        
        # Step 3: Train Unigram with EM
        self.train_unigram(text, tokens, characters)
        
        # Step 4: Save tokenizer
        self.save_tokenizer(tokens, characters)
        
        # Step 5: Test tokenizer
        test_sentences = [
            "Hello world!",
            "This is a test sentence.",
            "Xin chào thế giới!",
            "Đây là một câu thử nghiệm.",
            "Machine translation is amazing."
        ]
        self.test_tokenizer(test_sentences)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)


def main():
    """Main entry point"""
    
    # Configuration
    VOCAB_SIZE = 32000        # Target vocabulary size
    NUM_BPE_MERGES = 35000    # Number of BPE merge operations (must be > VOCAB_SIZE to allow pruning)
    MAX_SAMPLES = 500000      # Use 500k samples (None for all ~2.9M)
    OUTPUT_DIR = './tokenizer_models'
    
    # Create trainer
    trainer = PhomtTokenizerTrainer(
        vocab_size=VOCAB_SIZE,
        num_bpe_merges=NUM_BPE_MERGES,
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR
    )
    
    # Train tokenizer
    trainer.train()


if __name__ == '__main__':
    main()
