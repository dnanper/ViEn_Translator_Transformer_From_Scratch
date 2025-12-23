"""
Convert vocabulary.txt from 'token\tcount' format to 'id\ttoken' format
This script modifies the existing vocabulary file without retraining the tokenizer
"""

import sys
from pathlib import Path


def convert_vocabulary(vocab_path):
    """
    Convert vocabulary file from token\tcount to id\ttoken format
    
    Args:
        vocab_path: Path to vocabulary.txt file
    """
    vocab_path = Path(vocab_path)
    
    if not vocab_path.exists():
        print(f"❌ Error: File not found: {vocab_path}")
        sys.exit(1)
    
    print(f"📖 Reading vocabulary from: {vocab_path}")
    
    # Read existing vocabulary
    lines = []
    header_lines = []
    token_lines = []
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
            line_stripped = line.strip()
            
            # Keep header comments
            if line_stripped.startswith('#') or not line_stripped:
                header_lines.append(line)
            else:
                # Parse token\tcount format
                parts = line_stripped.split('\t')
                if len(parts) >= 1:
                    token = parts[0]
                    token_lines.append(token)
    
    print(f"✓ Found {len(token_lines)} tokens")
    
    # Create backup
    backup_path = vocab_path.with_suffix('.txt.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✓ Created backup: {backup_path}")
    
    # Write new format: id\ttoken
    with open(vocab_path, 'w', encoding='utf-8') as f:
        # Write updated header
        for line in header_lines:
            if line.strip().startswith('# Format:'):
                continue  # Skip old format line
            f.write(line)
        
        # Add new format comment
        f.write("# Format: ID\tTOKEN\n")
        f.write("\n")
        
        # Write tokens with IDs starting from 4 (after special tokens)
        token_id = 4
        for token in token_lines:
            f.write(f"{token_id}\t{token}\n")
            token_id += 1
    
    print(f"✓ Converted to id\ttoken format")
    print(f"✓ Saved: {vocab_path}")
    print(f"\n📊 Statistics:")
    print(f"   - Total tokens: {len(token_lines)}")
    print(f"   - ID range: 4 to {token_id - 1}")
    print(f"   - Special tokens (0-3): PAD, UNK, SOS, EOS")
    
    # Show sample
    print(f"\n📝 Sample (first 5 tokens):")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        skip_header = True
        count = 0
        for line in f:
            if skip_header:
                if line.strip() and not line.startswith('#'):
                    skip_header = False
                else:
                    continue
            
            if count < 5:
                print(f"   {line.rstrip()}")
                count += 1
            else:
                break


def main():
    """Main entry point"""
    
    # Default path
    default_vocab_path = Path(__file__).parent / 'tokenizer_models' / 'vocabulary.txt'
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        vocab_path = Path(sys.argv[1])
    else:
        vocab_path = default_vocab_path
    
    print("=" * 80)
    print("🔄 VOCABULARY FORMAT CONVERTER")
    print("=" * 80)
    print(f"\nConverting: token\\tcount → id\\ttoken\n")
    
    convert_vocabulary(vocab_path)
    
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
