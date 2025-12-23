# Vietnamese-English Translator

Complete Neural Machine Translation system using Transformer architecture.

## 📁 Project Structure

```
ViEn_Translator/
│
├── models_best/             # MODEL ARCHITECTURE ONLY
│   ├── __init__.py
│   ├── README.md
│   ├── config.py            # TransformerConfig
│   ├── transformer.py       # BestTransformer (main model)
│   ├── encoder.py           # Pre-LN encoder
│   ├── decoder.py           # Pre-LN decoder
│   ├── attention.py         # Multi-head attention
│   ├── feed_forward.py      # Feed-forward network
│   ├── embeddings.py        # Embedding layers
│   ├── positional_encoding.py  # Position encoding
│   ├── layer_norm.py        # LayerNorm & RMSNorm
│   ├── beam_search.py       # Beam search decoder
│   └── label_smoothing.py   # Label smoothing loss
│
├── trainer/                 # TRAINING & INFERENCE
│   ├── __init__.py
│   ├── train.py             # Training script (vi→en)
│   ├── inference.py         # Translation inference
│   └── evaluate.py          # BLEU evaluation
│
├── utils/                   # DATA PROCESSING
│   ├── __init__.py
│   └── data_processing.py   # DataProcessor, Dataset, collate_fn
│
├── SentencePiece-from-scratch/  #  TOKENIZER
│   ├── tokenizer_models/
│   │   ├── vocabulary.txt   # 32k vocab
│   │   └── metadata.txt
│   └── ...
│
├── data/                     # 📊 DATASETS
│   └── processed/
│       ├── train_tokenized.pkl
│       ├── validation_tokenized.pkl
│       └── test_tokenized.pkl
│
├── checkpoints/              # 💾 SAVED MODELS
│   ├── best_model_vi2en/
│   └── best_model_bidirectional/
│
├── config.py                 # Global config
└── README.md                 # This file
```

## ✨ Features

### Model Architecture

- **Pre-Layer Normalization** - More stable training
- **Weight Tying** - Decoder embedding = output projection
- **Label Smoothing** - Better generalization (0.1)
- **Beam Search** - High-quality inference with length penalty
- **Multi-Query Attention** - Faster inference (optional)
- **Gradient Clipping** - Prevent gradient explosion

### Training Features

- **Warmup LR Scheduler** - Linear warmup + inverse sqrt decay
- **Mixed Precision Training** - Faster with modern GPUs
- **Checkpoint Management** - Auto-save best model
- **Training Curves** - Automatic plotting
- **Resume Training** - Load from checkpoint

### Data Processing

- **SentencePiece Tokenizer** - 32,000 BPE tokens
- **Cached Tokenization** - Fast data loading (.pkl files)
- **Proper Masking** - Padding mask + causal mask
- **Bidirectional Support** - Train single model for both directions

## 🚀 Quick Start

### 0. Data preprocessing

```bash
python utils/data_processing.py
```

### 1. Train Vietnamese → English

```bash
python trainer/train.py
```

**Configuration:**

- Model: Base (512d, 6 layers, 65M params)
- Batch size: From `config.Config.BATCH_SIZE`
- Max length: From `config.Config.MAX_LEN`
- Device: Auto-detect CUDA/CPU
- Saves to: `checkpoints/best_model_vi2en/`

### 2. Inference (Translation)

#### a) Translate a file (batch, recommended)

```bash
python translate_test_file.py
```

This script translates `data/processed/test.en` to `data/processed/test_predict.vi` using the trained model. Output is also saved as a JSONL file for evaluation.

#### b) Fast inference from pre-tokenized data

```bash
python translate_from_tokenized.py --input data/processed/test_tokenized.pkl --output data/processed/test_predict.vi
```

See script for more options (batch size, device, etc). This is fastest for large test sets.

#### c) Translate a single sentence (interactive)

```python
from trainer.inference import Translator
translator = Translator(
    checkpoint_path='checkpoints/best_model_vi2en/best_model.pt',
    tokenizer_dir='SentencePiece-from-scratch/tokenizer_models',
    device='cuda'  # or 'cpu'
)
print(translator.translate("Xin chào thế giới"))
```

### 3. Evaluate BLEU Score

#### a) Evaluate from JSONL output (recommended)

```bash
python evaluate_jsonl.py
```

This compares the predicted translations in `data/processed/ep13_test_predict.jsonl` with the reference `data/processed/test.vi` and prints BLEU-1 to BLEU-4 scores. Results are saved as a JSON file.

#### b) Custom evaluation

You can modify `evaluate_jsonl.py` to point to your own prediction/reference files.

## 🛠️ Workflow Guide

### 1. Prepare Data

- **Train tokenizer:**

  ```bash
  cd SentencePiece-from-scratch
  python train_tokenizer_phomt.py
  ```

  Output: `tokenizer_models/` with vocabulary and model files.

- **Download & preprocess dataset:**

  ```python
  from utils.data_processing import DataProcessor
  processor = DataProcessor(config)
  processor.download_and_prepare_phomt()
  ```

  Output: `data/processed/` with train/validation/test splits in `.en`/`.vi` files.

- **Tokenize and cache datasets:**
  This is done automatically when running `train.py` for the first time.

### 2. Train Model

```bash
python trainer/train.py
```

Output: Model checkpoints in `checkpoints/best_model_vi2en/`.

### 3. Inference (Translation)

- **Batch translate test set:**

  ```bash
  python translate_test_file.py
  ```

  Output: `data/processed/test_predict.vi` and `ep13_test_predict.jsonl`.

- **(Optional) Fast batch inference:**
  ```bash
  python translate_from_tokenized.py --input data/processed/test_tokenized.pkl --output data/processed/test_predict.vi
  ```

### 4. Evaluation

```bash
python evaluate_jsonl.py
```

Output: BLEU scores and evaluation summary in `data/processed/ep13_evaluation_results.json`.

---

## 🔧 Advanced Usage

### Custom Configuration

```python
from models_best import TransformerConfig

config = TransformerConfig(
    d_model=512,
    n_encoder_layers=6,
    n_decoder_layers=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_len=512,
    learning_rate=1e-4,
    warmup_steps=8000,
    label_smoothing=0.1
)
```

## 🎯 Results

| Model | BLEU (en→vi) |
| ----- | ------------ |
| Base  | ~38          |

### Special Tokens

- PAD: 0
- UNK: 1
- SOS: 2 (Start of Sequence)
- EOS: 3 (End of Sequence)

## 📚 References

- **Attention Is All You Need** - Vaswani et al. (2017)
- **Pre-LN Transformer** - Xiong et al. (2020)
- **Label Smoothing** - Szegedy et al. (2016)
- **SentencePiece** - Kudo & Richardson (2018)

## 🤝 Contributing

Feel free to:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Translating! 🌍**
