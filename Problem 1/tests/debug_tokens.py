"""
Debug script to find invalid token IDs in dataset
Run this on Colab to identify the problem
"""

import sys
import pickle
from pathlib import Path

# Add to path
PROJECT_ROOT = Path('/content/ViEn_Translator_Transformer_From_Scratch')
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils.data_processing import DataProcessor

print("=" * 70)
print("Debugging Token IDs")
print("=" * 70)

# Load tokenizer
processor = DataProcessor(Config)
tokenizer_dir = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
processor.load_tokenizer(str(tokenizer_dir))

vocab_size = processor.vocab_size
print(f"\nVocab size: {vocab_size}")
print(f"Valid token range: [0, {vocab_size-1}]")

# Check cached data files
data_dir = PROJECT_ROOT / "data" / "processed"

for split in ['train', 'validation', 'test']:
    cache_file = data_dir / f"{split}_tokenized.pkl"
    
    if not cache_file.exists():
        print(f"\n{split}: File not found")
        continue
    
    print(f"\n{'='*70}")
    print(f"Checking {split} dataset...")
    print('='*70)
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    src_data = data['src']
    tgt_data = data['tgt']
    
    print(f"Total examples: {len(src_data)}")
    
    # Check for invalid IDs
    invalid_count = 0
    max_id_found = 0
    min_id_found = float('inf')
    
    for idx, (src_ids, tgt_ids) in enumerate(zip(src_data, tgt_data)):
        all_ids = src_ids + tgt_ids
        
        max_id = max(all_ids)
        min_id = min(all_ids)
        
        max_id_found = max(max_id_found, max_id)
        min_id_found = min(min_id_found, min_id)
        
        # Check if any ID is out of bounds
        if max_id >= vocab_size or min_id < 0:
            invalid_count += 1
            
            if invalid_count <= 5:  # Show first 5 errors
                print(f"\n❌ Example {idx}:")
                print(f"   Source IDs: {src_ids[:20]}... (len={len(src_ids)})")
                print(f"   Target IDs: {tgt_ids[:20]}... (len={len(tgt_ids)})")
                print(f"   Max ID: {max_id} (vocab_size={vocab_size})")
                print(f"   Min ID: {min_id}")
                
                if max_id >= vocab_size:
                    invalid_tokens = [i for i in all_ids if i >= vocab_size]
                    print(f"   Invalid tokens: {invalid_tokens[:10]}")
    
    print(f"\n📊 Statistics:")
    print(f"   Invalid examples: {invalid_count}/{len(src_data)}")
    print(f"   Max token ID found: {max_id_found}")
    print(f"   Min token ID found: {min_id_found}")
    print(f"   Vocab size: {vocab_size}")
    
    if max_id_found >= vocab_size:
        print(f"\n⚠️  PROBLEM: Max ID ({max_id_found}) >= vocab_size ({vocab_size})")
        print(f"   Tokens are out of bounds!")
    elif min_id_found < 0:
        print(f"\n⚠️  PROBLEM: Negative token IDs found!")
    else:
        print(f"\n✅ All token IDs are valid")

print("\n" + "=" * 70)
print("SOLUTION:")
print("=" * 70)

if max_id_found >= vocab_size:
    print("""
The problem is that tokenized data has IDs >= vocab_size.

This usually happens when:
1. Vocab size mismatch between tokenizer and model
2. Tokenizer metadata is wrong
3. Data was tokenized with different vocab

FIX:
1. Check vocabulary.txt first line for actual vocab size
2. Re-tokenize data if vocab changed
3. Or clip token IDs in dataloader (temporary fix)
""")
else:
    print("\n✅ No issues found with token IDs!")
