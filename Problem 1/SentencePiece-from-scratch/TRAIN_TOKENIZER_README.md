# Training Tokenizer on PhoMT Dataset

Hướng dẫn train và sử dụng SentencePiece tokenizer từ scratch trên dataset PhoMT (Vietnamese-English).

## 📋 Tổng Quan

Repository này implement tokenizer SentencePiece hoàn toàn từ đầu với **2-stage approach**:

1. **Stage 1 - Byte Pair Encoding (BPE)**: Tạo vocabulary ban đầu bằng cách gộp các cặp ký tự/token xuất hiện nhiều nhất
2. **Stage 2 - EM Algorithm với Unigram Probabilities**: Nhận tokens từ BPE, sau đó:
   - Tính unigram log-probabilities cho mỗi token
   - Dùng EM algorithm để tìm tokenization tối ưu
   - Pruning để giảm vocabulary về kích thước mong muốn
   - Hỗ trợ n-best sampling (stochastic tokenization)

**Lưu ý**: Đây KHÔNG phải là Unigram Language Model thuần túy như Google SentencePiece. Thay vào đó, nó kết hợp BPE để tạo vocabulary với EM để tối ưu tokenization.

## 🚀 Cài Đặt

### Yêu cầu

```bash
pip install datasets tqdm scipy numpy
```

### Cấu trúc files

```
SentencePiece-from-scratch/
├── byte_pair_encoder.py          # BPE implementation
├── sentence_piece.py              # Unigram LM với EM algorithm
├── train_tokenizer_phomt.py      # Script train tokenizer trên PhoMT
├── test_tokenizer.py              # Script test tokenizer
└── TRAIN_TOKENIZER_README.md     # File này
```

## 📚 Cách Sử Dụng

### 1. Train Tokenizer trên PhoMT

```bash
python train_tokenizer_phomt.py
```

**Cấu hình mặc định:**

- Vocabulary size: 32,000 tokens
- BPE merges: 10,000 operations
- Training samples: 500,000 pairs (có thể đổi thành `None` để dùng toàn bộ ~2.9M)
- Output directory: `./tokenizer_models/`

**Các bước thực hiện:**

1. **Download dataset** từ HuggingFace (`ura-hcmut/PhoMT`)
2. **Normalize và filter** dữ liệu
3. **Train BPE** - Gộp các cặp ký tự/token xuất hiện nhiều nhất
4. **Train Unigram LM** - Sử dụng EM algorithm để tối ưu vocabulary
5. **Pruning** - Giảm vocabulary về kích thước mong muốn
6. **Save models** - Lưu tokenizer và vocabulary

**Output files:**

```
tokenizer_models/
├── bpe_encoder.pkl              # BPE encoder model
├── sentencepiece_trainer.pkl    # SentencePiece trainer model
├── vocabulary.txt               # Vocabulary list (human-readable)
└── metadata.txt                 # Training metadata
```

### 2. Test Tokenizer

```bash
python test_tokenizer.py
```

Script này sẽ:

- Load tokenizer đã train
- Test trên câu tiếng Anh
- Test trên câu tiếng Việt
- Test trên câu mixed language
- Test edge cases
- So sánh n-best tokenizations
- Phân tích vocabulary statistics
- Chế độ interactive để test câu tùy ý

### 3. Sử dụng trong Code

```python
import pickle
from sentence_piece import SentencePieceTrainer

# Load tokenizer
with open('tokenizer_models/sentencepiece_trainer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Tokenize text
text = "Hello world! Xin chào thế giới!"
tokens = tokenizer.tokenize(text, nbest_size=1)
print(tokens)
# Output: ['Hello', '_world', '!', '_Xin', '_chào', '_thế', '_giới', '!']
```

## 🔧 Tùy Chỉnh Cấu Hình

Chỉnh sửa trong `train_tokenizer_phomt.py`:

```python
# Configuration
VOCAB_SIZE = 32000        # Kích thước vocabulary mong muốn
NUM_BPE_MERGES = 10000    # Số lượng BPE merge operations
MAX_SAMPLES = 500000      # Số lượng samples (None = toàn bộ)
OUTPUT_DIR = './tokenizer_models'

trainer = PhomtTokenizerTrainer(
    vocab_size=VOCAB_SIZE,
    num_bpe_merges=NUM_BPE_MERGES,
    max_samples=MAX_SAMPLES,
    output_dir=OUTPUT_DIR
)
```

### Gợi ý cấu hình:

| Use Case     | Vocab Size | BPE Merges | Samples     |
| ------------ | ---------- | ---------- | ----------- |
| Quick test   | 8,000      | 2,000      | 50,000      |
| Small model  | 16,000     | 5,000      | 200,000     |
| **Default**  | **32,000** | **10,000** | **500,000** |
| Large model  | 50,000     | 20,000     | 1,000,000   |
| Full dataset | 64,000     | 30,000     | None (all)  |

## 📊 Thuật Toán Chi Tiết

### Stage 1: Byte Pair Encoding (BPE)

```
1. Initialize: Mỗi ký tự là một token
2. Count: Đếm tần suất tất cả các bigrams
3. Merge: Gộp bigram xuất hiện nhiều nhất
4. Repeat: Lặp lại N lần (num_merges)
```

**Ví dụ:**

```
Input:  "hello world"
Step 0: h e l l o _ w o r l d _
Step 1: h e ll o _ w o r l d _      (merge l+l)
Step 2: h e ll o _ w o r ll d _     (merge l+l again)
Step 3: he ll o _ w o r ll d _      (merge h+e)
...
```

### Stage 2: EM Algorithm với Unigram Probabilities

**Quan trọng**: Đây không phải là Unigram LM thuần túy. Tokens đã được tạo sẵn từ BPE, và EM chỉ dùng để optimize probabilities và tìm tokenization tốt nhất.

```
E-step (Expectation):
  - Tokenize text với probabilities hiện tại
  - Đếm tần suất tokens trong tokenization
  - Update probabilities trong Trie

M-step (Maximization):
  - Tìm tokenization tối ưu bằng Viterbi (Dynamic Programming)
  - Maximize log-likelihood của tokenization

Pruning:
  - Xóa 20% tokens ít xuất hiện nhất (trừ base characters)
  - Lặp lại cho đến khi đạt vocab_size
```

**Forward Step (Viterbi):**

```python
# d[i] = max log-prob của tokenization cho text[:i]
# p[i] = độ dài token cuối cùng trong cách tokenize tốt nhất

for i in range(1, N+1):
    for j in range(max(i-maxlen, 0), i):
        token = text[j:i]
        if d[j] + prob(token) > d[i]:
            d[i] = d[j] + prob(token)
            p[i] = len(token)
```

## 🎯 Ví Dụ Output

### Tokenization Examples

```
Input:  "Machine learning is amazing."
Tokens: ['▁Machine', '▁learning', '▁is', '▁amazing', '.']

Input:  "Xin chào thế giới!"
Tokens: ['▁Xin', '▁chào', '▁thế', '▁giới', '!']

Input:  "I study at Đại học Bách Khoa."
Tokens: ['▁I', '▁study', '▁at', '▁Đại', '▁học', '▁Bách', '▁Khoa', '.']
```

(Lưu ý: `▁` là ký hiệu thay thế cho `_` để dễ nhìn, đại diện cho dấu cách)

### Top Frequent Tokens

```
1. '▁' → 2,456,789 occurrences
2. 'e' → 1,234,567 occurrences
3. 't' → 987,654 occurrences
4. '▁the' → 456,789 occurrences
5. 'ing' → 345,678 occurrences
...
```

## 🔍 So Sánh với SentencePiece Chính Thức

| Feature                   | Implementation này   | Google SentencePiece     |
| ------------------------- | -------------------- | ------------------------ |
| **Vocabulary Generation** | BPE only             | BPE hoặc Unigram LM      |
| **Tokenization**          | EM với unigram probs | BPE hoặc Unigram LM      |
| **Approach**              | 2-stage (BPE → EM)   | Single algorithm         |
| EM Algorithm              | ✅ Có (để optimize)  | ✅ Có (nếu dùng Unigram) |
| Pruning                   | ✅ Có                | ✅ Có                    |
| N-best sampling           | ✅ Có                | ✅ Có                    |
| Subword regularization    | ⚠️ Basic             | ✅ Advanced              |
| Character coverage        | ❌ Không             | ✅ Có                    |
| Performance               | 🐌 Chậm (Python)     | ⚡ Nhanh (C++)           |

**Điểm khác biệt chính**:

- Google SentencePiece cho phép chọn **hoặc** BPE **hoặc** Unigram LM
- Implementation này **kết hợp cả hai**: BPE tạo tokens, EM optimize tokenization

## 📈 Performance Notes

**Training time** (trên CPU, 500k samples):

- BPE: ~5-10 phút
- Unigram EM: ~10-20 phút
- Total: ~15-30 phút

**Memory usage:**

- ~2-4 GB RAM cho 500k samples
- ~8-16 GB RAM cho full dataset (~2.9M)

## 🐛 Troubleshooting

### Lỗi: Out of Memory

```python
# Giảm số lượng samples
MAX_SAMPLES = 100000  # Thay vì 500000
```

### Lỗi: Tokenization fails

```python
# Kiểm tra xem token có trong vocabulary không
# Có thể cần tăng num_bpe_merges hoặc giảm vocab_size
```

### Lỗi: Dataset download slow

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
```

## 📚 Tài Liệu Tham Khảo

1. **SentencePiece**: [GitHub Repository](https://github.com/google/sentencepiece)
2. **BPE Paper**: Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)
3. **Unigram LM**: Subword Regularization (Kudo, 2018)
4. **PhoMT Dataset**: [HuggingFace](https://huggingface.co/datasets/ura-hcmut/PhoMT)

## 🎓 Educational Purpose

Code này được viết với mục đích **học tập và hiểu sâu** về cách tokenizer hoạt động:

- ✅ Dễ đọc và có comment chi tiết
- ✅ Implement từ đầu, không dùng thư viện có sẵn
- ✅ Có visualization và logging rõ ràng
- ❌ Không tối ưu cho production (dùng Google SentencePiece thay thế)

## 📝 License

MIT License - Free to use for educational purposes

---

**Happy Tokenizing! 🚀**
