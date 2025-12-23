# Tests Directory

Folder này chứa các script test và debug cho project.

## 📝 Test Scripts

### 1. `test_setup.py`

**Mục đích:** Verify model configuration và data flow trước khi training

**Chạy:**

```bash
python tests/test_setup.py
```

**Tests:**

- ✅ Special token indices (PAD, UNK, SOS, EOS)
- ✅ Data shapes và batch processing
- ✅ Model forward pass
- ✅ Training step với loss calculation
- ✅ Beam search inference

---

### 2. `test_dataloader.py`

**Mục đích:** Test DataLoader với tokenizer đã train sẵn

**Chạy:**

```bash
python tests/test_dataloader.py
```

**Tests:**

- Load tokenizer
- Create dataloaders
- Sample batch inspection
- Token encoding/decoding

---

### 3. `test_direction.py`

**Mục đích:** Verify translation direction (EN→VI hoặc VI→EN)

**Chạy:**

```bash
python tests/test_direction.py
```

**Output:**

- Sample examples từ dataset
- Language detection (Vietnamese vs English)
- Kết luận hướng dịch

---

### 4. `test_paths.py`

**Mục đích:** Verify relative paths hoạt động đúng trên mọi môi trường

**Chạy:**

```bash
python tests/test_paths.py
```

**Tests:**

- Project root detection
- Directory structure
- Critical files existence
- Import paths

---

## 🐛 Debug Scripts

### 5. `debug_tokens.py`

**Mục đích:** Scan data để tìm invalid token IDs

**Chạy:**

```bash
python tests/debug_tokens.py
```

**Output:**

- Invalid token IDs (>= vocab_size hoặc < 0)
- Statistics về token distribution
- Examples có vấn đề

---

### 6. `fix_invalid_tokens.py`

**Mục đích:** Fix invalid token IDs trong cached data

**Chạy:**

```bash
python tests/fix_invalid_tokens.py
```

**Actions:**

- Replace token ID 32001 → 1 (UNK)
- Backup original files
- Verify sau khi fix

---

## 🚀 Quick Test Workflow

### Trước khi training:

```bash
# 1. Test paths
python tests/test_paths.py

# 2. Debug tokens
python tests/debug_tokens.py

# 3. Test setup
python tests/test_setup.py

# 4. Nếu test_setup pass → Ready to train!
python trainer/train.py
```

### Nếu gặp token ID issues:

```bash
# Run fix
python tests/fix_invalid_tokens.py

# Verify
python tests/debug_tokens.py

# Re-test
python tests/test_setup.py
```

---

## 📊 Expected Output

### ✅ All tests pass:

```
✅ PASS: Special token indices match
✅ PASS: Data shapes correct
✅ PASS: Model forward works
✅ PASS: Training step works
✅ PASS: Beam search works
```

### ❌ If tests fail:

- Check error messages
- Run debug_tokens.py
- Fix issues
- Re-run tests

---

## 🔧 Running on Different Environments

### Local:

```bash
python tests/test_setup.py
```

### Colab:

```python
!python tests/test_setup.py
```

### Kaggle:

```python
!python tests/test_setup.py
```

All paths are relative - works everywhere! ✅
