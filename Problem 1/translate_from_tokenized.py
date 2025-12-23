"""
Fast Inference from Pre-tokenized Data

Load tokenized IDs directly → DataLoader → Batch inference
Skips tokenization for maximum speed
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models_best import BestTransformer, TransformerConfig
from utils.data_processing import DataProcessor
from config import Config


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized sequences with optional source text"""
    
    def __init__(self, token_ids_list, source_texts=None):
        """
        Args:
            token_ids_list: List of tokenized sequences (list of list of ints)
            source_texts: Optional list of source text strings
        """
        self.data = token_ids_list
        self.source_texts = source_texts if source_texts is not None else [None] * len(token_ids_list)
        assert len(self.data) == len(self.source_texts)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'tokens': torch.tensor(self.data[idx], dtype=torch.long),
            'source': self.source_texts[idx],
            'idx': idx  # Keep original index for sorting
        }


def collate_fn(batch, pad_idx=0):
    """
    Collate function to pad sequences in batch
    
    Args:
        batch: List of dicts with 'tokens', 'source', 'idx'
        pad_idx: Padding index
        
    Returns:
        Dict with padded tokens, mask, sources, and indices
    """
    # Extract components
    tokens = [item['tokens'] for item in batch]
    sources = [item['source'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    # Pad sequences
    padded = pad_sequence(tokens, batch_first=True, padding_value=pad_idx)
    
    # Create mask
    mask = (padded != pad_idx).unsqueeze(1).unsqueeze(2)
    
    return {
        'tokens': padded,
        'mask': mask,
        'sources': sources,
        'indices': indices
    }


class FastTranslator:
    """Fast translator for pre-tokenized data"""
    
    def __init__(self, checkpoint_path: str, tokenizer_dir: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            tokenizer_dir: Directory containing tokenizer model
            device: Device to run on
        """
        print("Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        self.config.device = device
        
        # Load processor (for decoding only)
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
    
    def decode_tokens(self, token_ids):
        """Decode token IDs to text"""
        return self.processor.decode_sentence(token_ids, skip_special_tokens=True)
    
    def translate_batch_greedy(self, src_batch: torch.Tensor, max_len: int = 256):
        """
        Batch greedy decoding - TRUE PARALLEL BATCH PROCESSING
        
        Args:
            src_batch: Source batch [batch_size, src_len]
            max_len: Maximum decode length
            
        Returns:
            List of decoded token ID sequences
        """
        batch_size = src_batch.size(0)
        device = src_batch.device
        
        # Create source mask
        src_mask = (src_batch != self.processor.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Encode source batch ONCE (encoder handles embedding internally)
        memory = self.model.encoder(src_batch, src_mask)
        
        # Start with BOS token
        decoder_input = torch.full((batch_size, 1), self.config.bos_idx, 
                                   dtype=torch.long, device=device)
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            # Causal mask
            tgt_len = decoder_input.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).bool()
            
            # Decode step (decoder handles embedding internally)
            output = self.model.decoder(decoder_input, memory, src_mask, tgt_mask)
            
            # Greedy selection
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Mark finished
            finished = finished | (next_token.squeeze(1) == self.config.eos_idx)
            
            # Append
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Early stop
            if finished.all():
                break
        
        # Convert to list
        results = []
        for i in range(batch_size):
            token_ids = decoder_input[i].cpu().tolist()
            results.append(token_ids)
        
        return results
    
    def translate_dataloader(self, dataloader: DataLoader, output_file: str, 
                            save_jsonl: bool = True):
        """
        Translate entire dataloader with batch processing
        Maintains original order by tracking indices
        
        Args:
            dataloader: DataLoader with tokenized data
            output_file: Output file path
            save_jsonl: Save JSONL format
        """
        print(f"\nTranslating {len(dataloader.dataset)} samples...")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Output: {output_file}")
        
        output_path = Path(output_file)
        jsonl_path = output_path.parent / (output_path.stem + '.jsonl')
        
        # Collect all results first to sort by original order
        all_results = []
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Translating")):
                    # Extract batch components
                    src_batch = batch['tokens'].to(self.config.device)
                    sources = batch['sources']
                    indices = batch['indices']
                    
                    # Batch translate (TRUE PARALLEL)
                    batch_results = self.translate_batch_greedy(src_batch, max_len=256)
                    
                    # Collect results with original indices
                    for i, token_ids in enumerate(batch_results):
                        translation = self.decode_tokens(token_ids)
                        
                        all_results.append({
                            'idx': indices[i],
                            'source': sources[i],
                            'translation': translation
                        })
                        
                        # Print sample
                        if len(all_results) <= 5 or len(all_results) % 500 == 0:
                            src_preview = sources[i][:100] if sources[i] else "N/A"
                            print(f"\n[{len(all_results)}] EN: {src_preview}...")
                            print(f"      VI: {translation[:100]}...")
            
            # Sort by original index to maintain order
            all_results.sort(key=lambda x: x['idx'])
            
            print(f"\nWriting {len(all_results)} translations to files...")
            
            # Write sorted results
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for result in all_results:
                    f_out.write(result['translation'] + '\n')
            
            if save_jsonl:
                with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
                    for result in all_results:
                        f_jsonl.write(json.dumps({
                            'id': result['idx'],
                            'source': result['source'],
                            'translation': result['translation']
                        }, ensure_ascii=False) + '\n')
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        
        print(f"\n✓ Translated {len(all_results)} samples")
        print(f"✓ Output saved to:")
        print(f"  - Text: {output_file}")
        if save_jsonl:
            print(f"  - JSONL: {jsonl_path}")


def load_tokenized_data(file_path: str, source_file: str = None):
    """
    Load pre-tokenized data from pickle, text, or jsonl file
    
    Args:
        file_path: Path to tokenized file (.pkl, .txt, or .jsonl)
        source_file: Optional path to source text file (.en, .txt)
        
    Returns:
        Tuple of (token_ids_list, source_texts_list)
    """
    file_path = Path(file_path)
    
    # Load token IDs
    if file_path.suffix == '.pkl':
        # Load from pickle
        print(f"Loading tokenized data from {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(data, dict):
            token_ids = data.get('src', data.get('data', []))
        elif isinstance(data, list):
            token_ids = data
        else:
            raise ValueError(f"Unknown pickle format: {type(data)}")
        
        print(f"✓ Loaded {len(token_ids)} sequences")
        source_texts = None
    
    elif file_path.suffix == '.txt':
        # Load from text file (each line is space-separated token IDs)
        print(f"Loading tokenized data from {file_path}...")
        token_ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                ids = [int(x) for x in line.strip().split()]
                if ids:
                    token_ids.append(ids)
        
        print(f"✓ Loaded {len(token_ids)} sequences")
        source_texts = None
    
    elif file_path.suffix == '.jsonl':
        # Load from JSONL (with source text)
        print(f"Loading tokenized data from {file_path}...")
        token_ids = []
        source_texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Support different jsonl formats
                if 'tokens' in item or 'token_ids' in item:
                    ids = item.get('tokens', item.get('token_ids', []))
                    source = item.get('source', item.get('text', ''))
                else:
                    # If just has 'source' key, skip (no tokens)
                    continue
                
                token_ids.append(ids)
                source_texts.append(source)
        
        print(f"✓ Loaded {len(token_ids)} sequences with source text")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Load source text from separate file if provided
    if source_file and source_texts is None:
        print(f"Loading source text from {source_file}...")
        source_texts = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                source_texts.append(line.strip())
        print(f"✓ Loaded {len(source_texts)} source texts")
        
        # Verify lengths match
        if len(source_texts) != len(token_ids):
            print(f"⚠️  Warning: Source texts ({len(source_texts)}) != token sequences ({len(token_ids)})")
    
    return token_ids, source_texts


def main():
    """Main inference from tokenized data"""
    
    # Configuration
    PROJECT_ROOT = Path(__file__).parent
    
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_en2vi" / "checkpoint_epoch_13.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    
    # Input: Pre-tokenized data (change this to your tokenized file)
    TOKENIZED_FILE = PROJECT_ROOT / "data" / "processed" / "test_tokenized.pkl"  # or .txt or .jsonl
    
    # Optional: Separate source text file (if not in JSONL)
    SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "test.en"  # Optional
    # Set to None if not needed or if using JSONL with source
    SOURCE_FILE = None if TOKENIZED_FILE.suffix == '.jsonl' else SOURCE_FILE
    
    # Output
    OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ep13_test_predict.vi"
    
    # Settings
    BATCH_SIZE = 64  # Larger batch for speed (adjust based on GPU memory)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Fast Inference from Pre-tokenized Data")
    print("=" * 60)
    print(f"Tokens: {TOKENIZED_FILE}")
    if SOURCE_FILE:
        print(f"Source: {SOURCE_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Batch:  {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load tokenized data (with optional source texts)
    token_ids_list, source_texts = load_tokenized_data(
        str(TOKENIZED_FILE), 
        str(SOURCE_FILE) if SOURCE_FILE else None
    )
    
    # Create dataset and dataloader
    dataset = TokenizedDataset(token_ids_list, source_texts)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # IMPORTANT: Don't shuffle to maintain order
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        num_workers=0,  # 0 for debugging, increase for speed
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"\n✓ Created DataLoader:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Batches: {len(dataloader)}")
    print(f"  - Batch size: {BATCH_SIZE}")
    
    # Load model
    translator = FastTranslator(str(CHECKPOINT_PATH), str(TOKENIZER_DIR), DEVICE)
    
    # Translate
    translator.translate_dataloader(dataloader, str(OUTPUT_FILE), save_jsonl=True)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
