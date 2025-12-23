"""
Data Processing Module for English-Vietnamese NMT
Handles dataset downloading, cleaning, tokenization, and DataLoader creation
"""

import os
import re
import unicodedata
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Add SentencePiece-from-scratch to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SentencePiece-from-scratch'))
from sentence_piece import SentencePieceTrainer


class TranslationDataset(Dataset):
    """Dataset class for translation pairs"""
    
    def __init__(self, src_data: List[List[int]], tgt_data: List[List[int]]):
        """
        Args:
            src_data: List of tokenized source sentences (list of token IDs)
            tgt_data: List of tokenized target sentences (list of token IDs)
        """
        assert len(src_data) == len(tgt_data), "Source and target must have same length"
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_data[idx], dtype=torch.long),
            'tgt': torch.tensor(self.tgt_data[idx], dtype=torch.long)
        }


class DataProcessor:
    """Main data processing class"""
    
    # Special token IDs
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.pad_idx = self.PAD_IDX
        self.unk_idx = self.UNK_IDX
        self.sos_idx = self.SOS_IDX
        self.eos_idx = self.EOS_IDX
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: Unicode normalization, whitespace cleaning
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text
        """
        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_pair(self, src: str, tgt: str) -> bool:
        """
        Filter out bad sentence pairs based on heuristics
        
        Args:
            src: Source sentence
            tgt: Target sentence
            
        Returns:
            True if pair should be kept, False otherwise
        """
        # Length filtering
        src_len = len(src.split())
        tgt_len = len(tgt.split())
        
        # Remove empty sentences
        if src_len == 0 or tgt_len == 0:
            return False
        
        # Remove too long sentences
        if src_len > Config.MAX_LEN or tgt_len > Config.MAX_LEN:
            return False
        
        # Length ratio filtering
        length_ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
        if length_ratio > 1.5:
            return False
        
        return True
    
    def download_and_prepare_phomt(self):
        """
        Download and prepare PhoMT Vietnamese-English dataset (~2.9M pairs)
        Dataset: ura-hcmut/PhoMT (high-quality Vi-En parallel corpus)
        
        Returns:
            Dictionary with train, dev, test splits
        """
        print("Downloading PhoMT Vi-En dataset (~2.9M pairs)...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset("ura-hcmut/PhoMT")
        
        # Prepare data directory
        raw_dir = Path(Config.RAW_DATA_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = {
            'train': {'en': [], 'vi': []},
            'validation': {'en': [], 'vi': []},
            'test': {'en': [], 'vi': []},
            'private_test': {'en': [], 'vi': []}
        }
        
        # PhoMT has 'train', 'validation', and 'test' splits
        # We'll split validation into dev and test, and keep original test as private_test
        for split in ['train', 'validation', 'test']:
            print(f"\nProcessing {split} split...")
            split_data = dataset[split]
            
            # Map 'test' split to 'private_test' in our processed data
            target_split = 'private_test' if split == 'test' else split
            
            for example in tqdm(split_data):
                # PhoMT format: {'en': '...', 'vi': '...'}
                # Skip if either text is None or empty
                if example['en'] is None or example['vi'] is None:
                    continue
                if not example['en'].strip() or not example['vi'].strip():
                    continue
                
                en_text = self.normalize_text(example['en'])
                vi_text = self.normalize_text(example['vi'])
                
                # Filter bad pairs
                if self.filter_pair(en_text, vi_text):
                    processed_data[target_split]['en'].append(en_text)
                    processed_data[target_split]['vi'].append(vi_text)
        
        # Split validation into dev (50%) and test (50%)
        val_en = processed_data['validation']['en']
        val_vi = processed_data['validation']['vi']
        mid = len(val_en) // 2
        
        processed_data['test'] = {
            'en': val_en[:mid],
            'vi': val_vi[:mid]
        }
        processed_data['validation'] = {
            'en': val_en[mid:],
            'vi': val_vi[mid:]
        }
        
        # Save processed data
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'validation', 'test', 'private_test']:
            # Save English
            if split == 'private_test':
                with open(processed_dir / f"{split}.en", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(processed_data[split]['en']))
                
                # Save Vietnamese
                with open(processed_dir / f"{split}.vi", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(processed_data[split]['vi']))
        
        print(f"\nDataset statistics:")
        print(f"Train: {len(processed_data['train']['en'])} pairs")
        print(f"Validation: {len(processed_data['validation']['en'])} pairs")
        print(f"Test: {len(processed_data['test']['en'])} pairs")
        print(f"Private Test: {len(processed_data['private_test']['en'])} pairs")
        
        return processed_data
    

    
    def load_tokenizer(self, model_dir: str):
        """
        Load trained SentencePiece tokenizer and vocabulary
        
        Args:
            model_dir: Directory containing sentencepiece_trainer.pkl and vocabulary.txt
        """
        # Load tokenizer
        sp_path = os.path.join(model_dir, 'sentencepiece_trainer.pkl')
        with open(sp_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"   - Vocabulary size: {self.tokenizer.vocab_size:,}")
        print(f"   - Max token length: {self.tokenizer.maxlen}")
        
        # Load vocabulary mapping
        vocab_path = os.path.join(model_dir, 'vocabulary.txt')
        self.token2id = {}
        self.id2token = {}
        
        # Add special tokens first
        self.token2id['<PAD>'] = self.PAD_IDX
        self.token2id['<UNK>'] = self.UNK_IDX
        self.token2id['<SOS>'] = self.SOS_IDX
        self.token2id['<EOS>'] = self.EOS_IDX
        self.id2token[self.PAD_IDX] = '<PAD>'
        self.id2token[self.UNK_IDX] = '<UNK>'
        self.id2token[self.SOS_IDX] = '<SOS>'
        self.id2token[self.EOS_IDX] = '<EOS>'
        
        # Load tokens from vocabulary file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Format: id\ttoken
                parts = line.split('\t')
                if len(parts) >= 2:
                    token_id = int(parts[0])
                    token = parts[1]
                    self.token2id[token] = token_id
                    self.id2token[token_id] = token
        
        # Vocab size MUST be max_id + 1 to accommodate all token IDs
        # (not len(token2id) because IDs may have gaps or exceed count)
        max_id = max(self.id2token.keys()) if self.id2token else 3
        self.vocab_size = max_id + 1
        
        # Verify consistency
        token_count = len(self.token2id)
        if self.vocab_size != token_count:
            print(f"⚠️  Note: vocab_size ({self.vocab_size}) != token_count ({token_count})")
            print(f"   Using vocab_size = max_id + 1 = {max_id} + 1 = {self.vocab_size}")
        
        print(f"Loaded tokenizer from {model_dir}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Max token ID: {max_id}")
        print(f"Special tokens - PAD: {self.pad_idx}, UNK: {self.unk_idx}, "
              f"SOS: {self.sos_idx}, EOS: {self.eos_idx}")
    
    def encode_sentence(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode sentence to token IDs
        
        Args:
            text: Input text
            add_sos: Add SOS token
            add_eos: Add EOS token
            
        Returns:
            List of token IDs
        """
        # Tokenize with trained model (now handles unknown chars internally)
        tokens = self.tokenizer.tokenize(text, nbest_size=1)
        
        # Convert underscore to unicode block to match vocabulary format
        tokens = [t.replace('_', '▁') for t in tokens]
        # print(f"Tokens in encoder: {tokens}")
        
        # Convert tokens to IDs using vocab mapping
        # Unknown tokens get UNK_IDX - this is where UNK assignment happens
        token_ids = []
        
        if add_sos:
            token_ids.append(self.sos_idx)
        
        for token in tokens:
            token_id = self.token2id.get(token, self.unk_idx)
            token_ids.append(token_id)
        
        if add_eos:
            token_ids.append(self.eos_idx)
        
        # Truncate if too long
        if len(token_ids) > Config.MAX_LEN:
            token_ids = token_ids[:Config.MAX_LEN]
            if add_eos:
                token_ids[-1] = self.eos_idx
        
        return token_ids
    
    def decode_sentence(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens
            
        Returns:
            Decoded text
        """
        special_ids = {self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx}
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id2token.get(token_id, '<UNK>')
            tokens.append(token)
        
        # Join tokens and convert underscore back to space
        text = ''.join(tokens).replace('▁', ' ')
        return text.strip()
    
    def prepare_datasets(self) -> Dict[str, TranslationDataset]:
        """
        Load and prepare all datasets (train, validation, test)
        Cache tokenized IDs to avoid re-tokenizing every time
        
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        
        for split in ['train', 'validation', 'test', 'private_test']:
            # if split == 'train':
            # Check if cached tokenized IDs exist
            cache_file = processed_dir / f"{split}_tokenized.pkl"
            
            if cache_file.exists():
                print(f"\nLoading cached tokenized {split} dataset from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                src_encoded = cached_data['src']
                tgt_encoded = cached_data['tgt']
                print(f"✓ Loaded {len(src_encoded)} cached examples")
            else:
                print(f"\nPreparing {split} dataset (tokenizing and caching)...")
                
                # Read source and target files
                with open(processed_dir / f"{split}.en", 'r', encoding='utf-8') as f:
                    src_sentences = f.read().strip().split('\n')
                
                with open(processed_dir / f"{split}.vi", 'r', encoding='utf-8') as f:
                    tgt_sentences = f.read().strip().split('\n')
                
                # Encode sentences
                src_encoded = []
                tgt_encoded = []
                
                for src, tgt in tqdm(zip(src_sentences, tgt_sentences), total=len(src_sentences), desc="Tokenizing"):
                    src_ids = self.encode_sentence(src, add_sos=True, add_eos=True)
                    tgt_ids = self.encode_sentence(tgt, add_sos=True, add_eos=True)
                    
                    src_encoded.append(src_ids)
                    tgt_encoded.append(tgt_ids)
                
                # Cache tokenized IDs
                print(f"Saving tokenized data to {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump({'src': src_encoded, 'tgt': tgt_encoded}, f)
                print(f"✓ Cached {len(src_encoded)} examples")
            
            datasets[split] = TranslationDataset(src_encoded, tgt_encoded)
            print(f"{split}: {len(datasets[split])} examples")
        
        return datasets


def collate_fn(batch, pad_idx):
    """
    Custom collate function for DataLoader with dynamic batching
    
    Args:
        batch: List of examples from dataset
        pad_idx: Padding token index
        
    Returns:
        Batched and padded tensors with masks
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    # Create padding masks (1 for real tokens, 0 for padding)
    src_mask = (src_padded != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    tgt_mask = (tgt_padded != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    
    # Create look-ahead mask for decoder (causal mask)
    tgt_seq_len = tgt_padded.size(1)
    look_ahead_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len))).bool()  # [T, T]
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    
    # Combine padding mask and look-ahead mask for decoder
    tgt_mask = tgt_mask & look_ahead_mask
    
    return {
        'src': src_padded,      # [B, S]
        'tgt': tgt_padded,      # [B, T]
        'src_mask': src_mask,   # [B, 1, 1, S]
        'tgt_mask': tgt_mask    # [B, 1, T, T]
    }


def get_dataloaders(datasets: Dict[str, TranslationDataset], 
                   pad_idx: int,
                   batch_size: int = None) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits
    
    Args:
        datasets: Dictionary of datasets
        pad_idx: Padding token index
        batch_size: Batch size (default from config)
        
    Returns:
        Dictionary of DataLoaders
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    dataloaders = {}
    
    # Training DataLoader with shuffling
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if Config.USE_CUDA else False
    )
    
    # Validation and Test DataLoaders without shuffling
    for split in ['validation', 'test']:
        if split in datasets:
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, pad_idx),
                num_workers=0,
                pin_memory=True if Config.USE_CUDA else False
            )
    
    return dataloaders


if __name__ == "__main__":
    """Example usage"""
    
    # Initialize processor
    processor = DataProcessor(Config)
    
    # Download and prepare data (using PhoMT dataset)
    # Uncomment if you need to download the dataset
    data = processor.download_and_prepare_phomt()
    
    # Load pre-trained tokenizer from SentencePiece-from-scratch
    tokenizer_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "SentencePiece-from-scratch",
        "tokenizer_models"
    )
    processor.load_tokenizer(tokenizer_dir)
    
    # Prepare datasets
    datasets = processor.prepare_datasets()
    
    # Create dataloaders
    dataloaders = get_dataloaders(datasets, processor.pad_idx)
    
    print("\nDataLoader statistics:")
    for split, loader in dataloaders.items():
        print(f"{split}: {len(loader)} batches")
    
    # Test tokenization
    print("\n--- Testing tokenization ---")
    test_en = "Hello, how are you today?"
    test_vi = "Xin chào, bạn khỏe không?"
    
    print(f"\nEnglish: {test_en}")
    encoded_en = processor.encode_sentence(test_en)
    print(f"Encoded: {encoded_en}")
    decoded_en = processor.decode_sentence(encoded_en)
    print(f"Decoded: {decoded_en}")
    
    print(f"\nVietnamese: {test_vi}")
    encoded_vi = processor.encode_sentence(test_vi)
    print(f"Encoded: {encoded_vi}")
    decoded_vi = processor.decode_sentence(encoded_vi)
    print(f"Decoded: {decoded_vi}")
