# 🐛 Fix CUDA Index Out of Bounds Error

## ❌ Lỗi gặp phải:

```
CUDA error: device-side assert triggered
vectorized_gather_kernel: index out of bounds
```

## 🔍 Nguyên nhân:

Token IDs trong data **>= vocab_size** → Embedding layer không thể lookup

Ví dụ:

- Vocab size = 32,001 (model biết)
- Token ID = 32,001 hoặc lớn hơn (trong data)
- → Index out of bounds!

---

## ✅ Cách Fix Trên Colab

### Bước 1: Chạy script debug

```python
# Cell 1: Debug token IDs
!cd /content/ViEn_Translator_Transformer_From_Scratch && python debug_tokens.py
```

Script này sẽ:

- Scan toàn bộ data
- Tìm max token ID
- So sánh với vocab_size
- Báo cáo nếu có vấn đề

### Bước 2: Chạy quick fix

```python
# Cell 2: Apply quick fix
!cd /content/ViEn_Translator_Transformer_From_Scratch && python colab_quick_fix.py
```

Script này sẽ cho bạn biết:

- Vocab size đúng là bao nhiêu
- Cần sửa gì trong train.py

### Bước 3: Patch code trực tiếp

```python
# Cell 3: Patch train.py
import re

file_path = '/content/ViEn_Translator_Transformer_From_Scratch/trainer/train.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find actual vocab size from debug output
# Let's say it's 32001 (adjust based on colab_quick_fix.py output)
ACTUAL_VOCAB_SIZE = 32001  # ← THAY ĐỔI NẾU CẦN

# Option A: Fix by clamping token IDs (safe but slower)
old_line = "            # Validate token IDs (防止 index out of bounds)"
new_code = f"""            # Validate token IDs (防止 index out of bounds)
            vocab_size = {ACTUAL_VOCAB_SIZE}
            src = torch.clamp(src, 0, vocab_size - 1)
            tgt = torch.clamp(tgt, 0, vocab_size - 1)"""

if old_line not in content:
    # Add the validation code
    search = "            src = batch['src'].to(self.config.device)"
    replace = f\"\"\"            src = batch['src'].to(self.config.device)
            tgt_original = batch['tgt'].to(self.config.device)

            # Validate token IDs
            vocab_size = {ACTUAL_VOCAB_SIZE}
            src = torch.clamp(src, 0, vocab_size - 1)
            tgt = torch.clamp(tgt_original, 0, vocab_size - 1)\"\"\"

    content = content.replace(search, replace)
    content = content.replace(
        "tgt = batch['tgt'].to(self.config.device)",
        "# tgt moved above with validation"
    )

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Patched train.py with token ID validation")
```

---

## 🔧 Giải Pháp Chi Tiết

### Giải pháp 1: Clamp token IDs (Nhanh)

Thêm vào `train_epoch()` trong `trainer/train.py`:

```python
# After loading batch to device
src = batch['src'].to(self.config.device)
tgt = batch['tgt'].to(self.config.device)

# Clamp to valid range
vocab_size = self.model.src_vocab_size
src = torch.clamp(src, 0, vocab_size - 1)
tgt = torch.clamp(tgt, 0, vocab_size - 1)
```

**Ưu điểm:** Fix ngay lập tức  
**Nhược điểm:** Không giải quyết nguyên nhân gốc

### Giải pháp 2: Fix vocab_size (Đúng đắn)

1. **Kiểm tra vocab thật sự:**

```python
# Count lines in vocabulary.txt
!wc -l /content/ViEn_Translator_Transformer_From_Scratch/SentencePiece-from-scratch/tokenizer_models/vocabulary.txt
```

2. **Update metadata.txt:**

```python
# Update vocab_size in metadata
metadata_file = '/content/ViEn_Translator_Transformer_From_Scratch/SentencePiece-from-scratch/tokenizer_models/metadata.txt'

with open(metadata_file, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('vocab_size'):
        new_lines.append(f'vocab_size = 32001\n')  # Or actual vocab size
    else:
        new_lines.append(line)

with open(metadata_file, 'w') as f:
    f.writelines(new_lines)

print("✅ Updated metadata.txt")
```

3. **Reload tokenizer:**

```python
# Restart Python kernel and re-run training
# Or manually reload processor
```

### Giải pháp 3: Re-tokenize data (Triệt để)

Nếu data bị corrupt:

```python
from utils.data_processing import DataProcessor
from config import Config
import os

processor = DataProcessor(Config)
tokenizer_dir = '/content/ViEn_Translator_Transformer_From_Scratch/SentencePiece-from-scratch/tokenizer_models'
processor.load_tokenizer(tokenizer_dir)

# Delete old cached files
import os
data_dir = '/content/ViEn_Translator_Transformer_From_Scratch/data/processed'
for f in ['train_tokenized.pkl', 'validation_tokenized.pkl', 'test_tokenized.pkl']:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {f}")

# Re-tokenize
datasets = processor.prepare_datasets()
print("✅ Re-tokenized all data")
```

---

## 📝 Temporary Fix (Để train ngay)

Copy đoạn code này vào cell trước khi train:

```python
# EMERGENCY FIX - Run this before training
import torch

# Monkey patch Embedding to clamp indices
original_embedding_forward = torch.nn.Embedding.forward

def safe_embedding_forward(self, input):
    # Clamp input to valid range
    input = torch.clamp(input, 0, self.num_embeddings - 1)
    return original_embedding_forward(self, input)

torch.nn.Embedding.forward = safe_embedding_forward
print("✅ Patched Embedding layer to auto-clamp indices")

# Now run training
!python trainer/train.py
```

⚠️ **Warning:** Đây chỉ là quick fix! Nên tìm và fix nguyên nhân gốc.

---

## 🎯 Checklist

- [ ] Chạy `debug_tokens.py` để tìm max token ID
- [ ] Chạy `colab_quick_fix.py` để xác định vocab_size đúng
- [ ] So sánh:
  - Vocab size in metadata: `___`
  - Actual vocab lines: `___`
  - Max token ID in data: `___`
- [ ] Apply fix (chọn 1):
  - [ ] Clamp token IDs (temporary)
  - [ ] Update metadata.txt (proper)
  - [ ] Re-tokenize data (thorough)
- [ ] Re-run training

---

## 💡 Prevent Future Issues

Trong `data_processing.py`, thêm validation:

```python
def encode_sentence(self, text, add_sos=False, add_eos=False):
    # ... existing code ...

    # Validate token IDs
    max_id = max(token_ids) if token_ids else 0
    if max_id >= self.vocab_size:
        print(f"⚠️  Warning: Token ID {max_id} >= vocab_size {self.vocab_size}")
        # Clip to valid range
        token_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]

    return token_ids
```

---

Chúc bạn fix bug thành công! 🚀
