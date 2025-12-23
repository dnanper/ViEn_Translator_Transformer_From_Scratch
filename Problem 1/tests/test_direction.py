"""
Quick test to verify translation direction: EN → VI or VI → EN
"""

import sys
import os
from pathlib import Path

# Add parent to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processing import DataProcessor
from config import Config

def main():
    print("=" * 70)
    print("Verifying Translation Direction")
    print("=" * 70)
    
    # Initialize processor
    processor = DataProcessor(Config)
    
    # Load tokenizer
    tokenizer_dir = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    print(f"\nLoading tokenizer from: {tokenizer_dir}")
    processor.load_tokenizer(str(tokenizer_dir))
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = processor.prepare_datasets()
    
    # Check first few examples from train set
    print("\n" + "-" * 70)
    print("Sample Examples from Training Set:")
    print("-" * 70)
    
    num_samples = 5
    for i in range(min(num_samples, len(datasets['train']))):
        example = datasets['train'][i]
        
        src_tokens = example['src'].tolist()
        tgt_tokens = example['tgt'].tolist()
        
        src_text = processor.decode_sentence(src_tokens)
        tgt_text = processor.decode_sentence(tgt_tokens)
        
        print(f"\nExample {i+1}:")
        print(f"  Source: {src_text[:100]}...")
        print(f"  Target: {tgt_text[:100]}...")
    
    # Analyze language
    print("\n" + "=" * 70)
    print("Language Detection (Simple Heuristic):")
    print("=" * 70)
    
    # Sample 100 examples
    sample_size = min(100, len(datasets['train']))
    
    src_has_vietnamese = 0
    src_has_english = 0
    tgt_has_vietnamese = 0
    tgt_has_english = 0
    
    # Vietnamese characters
    vi_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
    
    for i in range(sample_size):
        example = datasets['train'][i]
        
        src_text = processor.decode_sentence(example['src'].tolist()).lower()
        tgt_text = processor.decode_sentence(example['tgt'].tolist()).lower()
        
        # Check Vietnamese characters
        if any(c in vi_chars for c in src_text):
            src_has_vietnamese += 1
        if any(c in vi_chars for c in tgt_text):
            tgt_has_vietnamese += 1
        
        # Check common English words
        en_words = {'the', 'is', 'are', 'and', 'or', 'to', 'of', 'in', 'for', 'on'}
        if any(word in src_text.split() for word in en_words):
            src_has_english += 1
        if any(word in tgt_text.split() for word in en_words):
            tgt_has_english += 1
    
    print(f"\nAnalyzed {sample_size} examples:")
    print(f"\nSource:")
    print(f"  Contains Vietnamese chars: {src_has_vietnamese}/{sample_size} ({src_has_vietnamese/sample_size*100:.1f}%)")
    print(f"  Contains English words:    {src_has_english}/{sample_size} ({src_has_english/sample_size*100:.1f}%)")
    
    print(f"\nTarget:")
    print(f"  Contains Vietnamese chars: {tgt_has_vietnamese}/{sample_size} ({tgt_has_vietnamese/sample_size*100:.1f}%)")
    print(f"  Contains English words:    {tgt_has_english}/{sample_size} ({tgt_has_english/sample_size*100:.1f}%)")
    
    # Determine direction
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    if src_has_english > src_has_vietnamese and tgt_has_vietnamese > tgt_has_english:
        print("\n✅ Translation Direction: ENGLISH → VIETNAMESE")
        print("   Source: English")
        print("   Target: Vietnamese")
    elif src_has_vietnamese > src_has_english and tgt_has_english > tgt_has_vietnamese:
        print("\n✅ Translation Direction: VIETNAMESE → ENGLISH")
        print("   Source: Vietnamese")
        print("   Target: English")
    else:
        print("\n⚠️  Cannot determine direction clearly!")
        print("   Please check data manually")
    
    print("\n" + "=" * 70)
    
    # Check data files directly
    print("\nVerifying raw data files:")
    print("-" * 70)
    
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    # Read first line from each file
    files_to_check = [
        ("train.en", "English Training"),
        ("train.vi", "Vietnamese Training"),
        ("validation.en", "English Validation"),
        ("validation.vi", "Vietnamese Validation"),
    ]
    
    for filename, label in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            print(f"\n{label} ({filename}):")
            print(f"  {first_line[:100]}...")
        else:
            print(f"\n{label} ({filename}): NOT FOUND")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
