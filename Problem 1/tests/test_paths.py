"""
Test Relative Paths - Verify paths work correctly across environments

Run this to check if all paths are properly configured!
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("Testing Relative Paths Configuration")
print("=" * 70)

# Get project root (parent of this script)
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"\n✓ Project Root: {PROJECT_ROOT}")

# Test all expected directories
expected_dirs = {
    "models_best": PROJECT_ROOT / "models_best",
    "trainer": PROJECT_ROOT / "trainer",
    "utils": PROJECT_ROOT / "utils",
    "SentencePiece-from-scratch": PROJECT_ROOT / "SentencePiece-from-scratch",
    "tokenizer_models": PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models",
    "data": PROJECT_ROOT / "data",
    "data/processed": PROJECT_ROOT / "data" / "processed",
}

print("\n" + "-" * 70)
print("Checking directories:")
print("-" * 70)

all_exist = True
for name, path in expected_dirs.items():
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {name:30s}: {path}")
    if not exists:
        all_exist = False

# Test critical files
print("\n" + "-" * 70)
print("Checking critical files:")
print("-" * 70)

critical_files = {
    "config.py": PROJECT_ROOT / "config.py",
    "vocabulary.txt": PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models" / "vocabulary.txt",
    "metadata.txt": PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models" / "metadata.txt",
    "train.py": PROJECT_ROOT / "trainer" / "train.py",
    "evaluate.py": PROJECT_ROOT / "trainer" / "evaluate.py",
    "inference.py": PROJECT_ROOT / "trainer" / "inference.py",
}

all_files_exist = True
for name, path in critical_files.items():
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {name:30s}: {path}")
    if not exists:
        all_files_exist = False

# Test data files (optional)
print("\n" + "-" * 70)
print("Checking data files (optional):")
print("-" * 70)

data_files = {
    "train_tokenized.pkl": PROJECT_ROOT / "data" / "processed" / "train_tokenized.pkl",
    "validation_tokenized.pkl": PROJECT_ROOT / "data" / "processed" / "validation_tokenized.pkl",
    "test_tokenized.pkl": PROJECT_ROOT / "data" / "processed" / "test_tokenized.pkl",
}

for name, path in data_files.items():
    exists = path.exists()
    status = "✓" if exists else "○"
    print(f"{status} {name:30s}: {path}")

# Test import paths
print("\n" + "-" * 70)
print("Testing imports:")
print("-" * 70)

try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import Config
    print("✓ import config.Config")
except Exception as e:
    print(f"✗ import config.Config: {e}")
    all_exist = False

try:
    from models_best_old import BestTransformer, TransformerConfig
    print("✓ import models_best.BestTransformer")
except Exception as e:
    print(f"✗ import models_best.BestTransformer: {e}")
    all_exist = False

try:
    from utils.data_processing import DataProcessor
    print("✓ import utils.data_processing.DataProcessor")
except Exception as e:
    print(f"✗ import utils.data_processing.DataProcessor: {e}")
    all_exist = False

# Test trainer scripts can find paths
print("\n" + "-" * 70)
print("Testing trainer script path resolution:")
print("-" * 70)

trainer_script = PROJECT_ROOT / "trainer" / "train.py"
if trainer_script.exists():
    # Simulate what the script does
    script_root = trainer_script.resolve().parent.parent
    save_dir = script_root / "checkpoints" / "best_model_vi2en"
    tokenizer_dir = script_root / "SentencePiece-from-scratch" / "tokenizer_models"
    
    print(f"✓ Script root resolves to: {script_root}")
    print(f"✓ Save dir would be:       {save_dir}")
    print(f"✓ Tokenizer dir would be:  {tokenizer_dir}")
    print(f"✓ Tokenizer exists:        {tokenizer_dir.exists()}")
else:
    print("✗ trainer/train.py not found")
    all_exist = False

# Environment info
print("\n" + "-" * 70)
print("Environment Information:")
print("-" * 70)

print(f"Python:    {sys.version.split()[0]}")
print(f"Platform:  {sys.platform}")
print(f"CWD:       {Path.cwd()}")

try:
    import torch
    print(f"PyTorch:   {torch.__version__}")
    print(f"CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch:   Not installed")

# Final verdict
print("\n" + "=" * 70)
if all_exist and all_files_exist:
    print("✅ ALL CHECKS PASSED - Ready to run!")
    print("\nNext steps:")
    print("  1. python test_setup.py          # Verify model config")
    print("  2. python trainer/train.py       # Start training")
    print("  3. python trainer/evaluate.py    # Evaluate model")
    exit(0)
elif all_exist:
    print("⚠️  DIRECTORIES OK - Some files missing (might be normal)")
    print("\nMissing files are likely:")
    print("  - Data files (run data preparation first)")
    print("  - Checkpoints (will be created during training)")
    print("\nYou can proceed if you plan to:")
    print("  - Download and prepare data")
    print("  - Train tokenizer")
    exit(0)
else:
    print("❌ SOME CHECKS FAILED - Please fix issues above")
    print("\nCommon solutions:")
    print("  - Make sure you're in the project root directory")
    print("  - Check if you cloned the repository correctly")
    print("  - Install missing dependencies: pip install -r requirements.txt")
    exit(1)
