"""
Script to translate test.en file to test_predict.vi
"""

import torch
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from trainer.inference import Translator

def main():
    # Get project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Paths
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_en2vi" / "best_model.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "test.en"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "test_predict.vi"
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("English → Vietnamese Translation")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"\n❌ Error: Input file not found: {INPUT_FILE}")
        return
    
    # Check if checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"\n❌ Error: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    # Create translator
    translator = Translator(
        checkpoint_path=str(CHECKPOINT_PATH),
        tokenizer_dir=str(TOKENIZER_DIR),
        device=DEVICE
    )
    
    # Translate file with batching
    translator.translate_file(
        input_file=str(INPUT_FILE),
        output_file=str(OUTPUT_FILE),
        beam_size=1,
        use_greedy=True,
        batch_size=256,
        save_jsonl=True  # Save JSONL for incremental results
    )
    
    print("\n" + "=" * 60)
    print("✓ Translation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
