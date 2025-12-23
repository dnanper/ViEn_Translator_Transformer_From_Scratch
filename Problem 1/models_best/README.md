# Best Transformer Model for Neural Machine Translation

## Các Cải Tiến So Với Implementation Gốc

### 1. **Pre-Layer Normalization (Pre-LN)**

- Ổn định training hơn Post-LN
- Gradient flow tốt hơn
- Ít cần warmup steps

### 2. **Label Smoothing**

- Giảm overfitting
- Cải thiện generalization

### 3. **Warmup Learning Rate Scheduler**

- Tăng LR tuyến tính trong warmup steps
- Decay theo inverse square root sau đó

### 4. **Dropout Variants**

- Attention dropout
- FFN dropout
- Residual dropout

### 5. **Weight Tying**

- Tie decoder embedding với output projection
- Giảm parameters, tăng performance

### 6. **Scaled Initialization**

- Xavier/Kaiming initialization
- Residual scaling theo số layers

### 7. **Gradient Clipping**

- Ngăn gradient explosion
- Ổn định training

### 8. **Beam Search với Length Penalty**

- Inference chất lượng cao hơn greedy
- Length normalization để tránh bias ngắn

### 9. **Multi-Head Attention với Relative Position**

- Optional relative position bias
- Tốt hơn cho long sequences

### 10. **Mixed Precision Training Support**

- Faster training với GPU hiện đại
- Giảm memory usage

## Cấu Trúc

```
models_best/               # Model Architecture ONLY
├── __init__.py
├── README.md
├── attention.py          # Multi-head attention
├── encoder.py           # Pre-LN encoder
├── decoder.py           # Pre-LN decoder
├── feed_forward.py      # FFN with dropout
├── positional_encoding.py  # Sinusoidal PE
├── layer_norm.py        # LayerNorm & RMSNorm
├── embeddings.py        # Embedding layers
├── transformer.py       # Complete model
├── beam_search.py       # Beam search decoder
├── label_smoothing.py   # Label smoothing loss
└── config.py           # Model configurations

trainer/                  # Training & Inference
├── __init__.py
├── train.py             # Training script (vi→en)
├── train_bidirectional.py  # Bidirectional training (vi↔en)
├── inference.py         # Translation inference
└── evaluate.py          # BLEU evaluation

utils/                    # Data Processing
├── __init__.py
└── data_processing.py   # DataProcessor, Dataset, collate_fn

```

## Hyperparameters Tốt Nhất cho Vietnamese-English

### Small Model (Fast training, ~60M params)

- d_model: 256
- n_layers: 4
- n_heads: 8
- d_ff: 1024
- dropout: 0.3
- label_smoothing: 0.1

### Base Model (Balanced, ~65M params)

- d_model: 512
- n_layers: 6
- n_heads: 8
- d_ff: 2048
- dropout: 0.1
- label_smoothing: 0.1

### Large Model (Best quality, ~213M params)

- d_model: 1024
- n_layers: 6
- n_heads: 16
- d_ff: 4096
- dropout: 0.3
- label_smoothing: 0.1

## Training Tips

1. **Learning Rate**:

   - Warmup: 4000-8000 steps
   - Peak LR: 1e-4 to 5e-4
   - Decay: inverse_sqrt

2. **Batch Size**:

   - Tối thiểu: 4096 tokens/batch
   - Tối ưu: 25000-32000 tokens/batch
   - Gradient accumulation nếu GPU nhỏ

3. **Regularization**:

   - Dropout: 0.1-0.3 (cao hơn nếu data ít)
   - Label smoothing: 0.1
   - Weight decay: 1e-4

4. **Optimization**:
   - Adam with β1=0.9, β2=0.98, ε=1e-9
   - Gradient clipping: max_norm=1.0

## Usage

```python
from models_best import BestTransformer
from models_best.config import TransformerConfig

# Load config
config = TransformerConfig.base()  # or .small(), .large()

# Create model
model = BestTransformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    config=config
)

# Train
model.train()
output = model(src, tgt, src_mask, tgt_mask)

# Inference with beam search
model.eval()
translation = model.translate_beam(
    src_sentence,
    beam_size=5,
    max_len=100,
    length_penalty=0.6
)
```

## References

1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
3. "Understanding the Difficulty of Training Transformers" (Liu et al., 2020)
4. "Training Tips for the Transformer Model" (Popel & Bojar, 2018)
