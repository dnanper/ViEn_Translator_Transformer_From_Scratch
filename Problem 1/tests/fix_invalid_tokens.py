"""
Fix invalid token IDs in cached tokenized data
Replaces token ID 32001 (out of bounds) with 1 (UNK token)
"""

import pickle
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path('/content/ViEn_Translator_Transformer_From_Scratch')
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("Fixing Invalid Token IDs")
print("=" * 70)

VOCAB_SIZE = 32001
UNK_ID = 1
INVALID_ID = 32001

data_dir = PROJECT_ROOT / "data" / "processed"

for split in ['train', 'validation', 'test']:
    cache_file = data_dir / f"{split}_tokenized.pkl"
    
    if not cache_file.exists():
        print(f"\n{split}: File not found, skipping...")
        continue
    
    print(f"\n{'='*70}")
    print(f"Processing {split} dataset...")
    print('='*70)
    
    # Load data
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    src_data = data['src']
    tgt_data = data['tgt']
    
    print(f"Total examples: {len(src_data)}")
    
    # Fix invalid IDs
    fixed_count = 0
    fixed_examples = []
    
    for idx, (src_ids, tgt_ids) in enumerate(zip(src_data, tgt_data)):
        src_fixed = False
        tgt_fixed = False
        
        # Check and fix source
        if any(tid >= VOCAB_SIZE or tid < 0 for tid in src_ids):
            src_data[idx] = [min(max(tid, 0), VOCAB_SIZE - 1) if tid != INVALID_ID else UNK_ID 
                             for tid in src_ids]
            src_fixed = True
        
        # Check and fix target
        if any(tid >= VOCAB_SIZE or tid < 0 for tid in tgt_ids):
            tgt_data[idx] = [min(max(tid, 0), VOCAB_SIZE - 1) if tid != INVALID_ID else UNK_ID 
                             for tid in tgt_ids]
            tgt_fixed = True
        
        if src_fixed or tgt_fixed:
            fixed_count += 1
            if fixed_count <= 5:
                fixed_examples.append(idx)
                print(f"\n✓ Fixed example {idx}:")
                if src_fixed:
                    print(f"  Source: {tgt_ids[:20]}... → {src_data[idx][:20]}...")
                if tgt_fixed:
                    print(f"  Target: {tgt_ids[:20]}... → {tgt_data[idx][:20]}...")
    
    print(f"\n📊 Summary:")
    print(f"  Fixed examples: {fixed_count}/{len(src_data)}")
    print(f"  Changed: {INVALID_ID} → {UNK_ID} (UNK)")
    
    if fixed_count > 0:
        # Backup original file
        backup_file = data_dir / f"{split}_tokenized.pkl.backup"
        import shutil
        shutil.copy2(cache_file, backup_file)
        print(f"  Backup saved: {backup_file.name}")
        
        # Save fixed data
        with open(cache_file, 'wb') as f:
            pickle.dump({'src': src_data, 'tgt': tgt_data}, f)
        print(f"  ✅ Saved fixed data to {cache_file.name}")
    else:
        print(f"  ✅ No fixes needed")

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Verify all IDs are now valid
for split in ['train', 'validation', 'test']:
    cache_file = data_dir / f"{split}_tokenized.pkl"
    
    if not cache_file.exists():
        continue
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    src_data = data['src']
    tgt_data = data['tgt']
    
    max_id = 0
    min_id = VOCAB_SIZE
    
    for src_ids, tgt_ids in zip(src_data, tgt_data):
        all_ids = src_ids + tgt_ids
        max_id = max(max_id, max(all_ids))
        min_id = min(min_id, min(all_ids))
    
    status = "✅" if max_id < VOCAB_SIZE and min_id >= 0 else "❌"
    print(f"{status} {split}: min={min_id}, max={max_id}, valid_range=[0, {VOCAB_SIZE-1}]")

print("\n" + "=" * 70)
print("✅ DONE! You can now run training:")
print("   python trainer/train.py")
print("=" * 70)
