"""
Quick test to verify vocab_size fix on Colab
Run this BEFORE training to ensure vocab_size = 32002
"""

import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path('/content/ViEn_Translator_Transformer_From_Scratch')
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("VOCAB SIZE FIX VERIFICATION")
print("=" * 70)

from config import Config
from utils.data_processing import DataProcessor

# Load tokenizer
processor = DataProcessor(Config)
tokenizer_dir = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
processor.load_tokenizer(str(tokenizer_dir))

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)

# Get actual max ID from vocabulary
with open(tokenizer_dir / "vocabulary.txt", 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    token_ids = [int(l.split('\t')[0]) for l in lines if '\t' in l]
    actual_max_id = max(token_ids)

print(f"Max token ID in vocabulary.txt: {actual_max_id}")
print(f"Processor vocab_size:           {processor.vocab_size}")
print(f"Expected vocab_size:            {actual_max_id + 1}")

print("\n" + "=" * 70)

if processor.vocab_size == actual_max_id + 1:
    print("✅ SUCCESS: vocab_size is CORRECT!")
    print(f"   Model will create Embedding[{processor.vocab_size}]")
    print(f"   Token IDs [0, {actual_max_id}] can be safely used")
    print("\n🚀 Ready to train!")
    exit(0)
else:
    print("❌ FAILED: vocab_size is WRONG!")
    print(f"   Expected: {actual_max_id + 1}")
    print(f"   Got:      {processor.vocab_size}")
    print(f"   Difference: {(actual_max_id + 1) - processor.vocab_size}")
    print("\n⚠️  Training will fail with 'index out of bounds' error!")
    print("\nDEBUG INFO:")
    print(f"  - Check if utils/data_processing.py has latest changes")
    print(f"  - Ensure git pull was successful")
    print(f"  - Try restarting Python kernel")
    exit(1)
