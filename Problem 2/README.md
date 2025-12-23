# README for t2_p1.py

This script trains a large language model (LLM) for English-Vietnamese medical translation using LoRA and streaming data loading. It is optimized for high-end GPUs (e.g., RTX 5090) and large models (Qwen/Qwen3-1.7B or larger).

## Features

- **Zero-RAM Streaming Dataset:** Loads data line-by-line, suitable for very large datasets.
- **Bidirectional Training:** Trains both English→Vietnamese and Vietnamese→English directions.
- **Dynamic Prompting:** Mixes instructional and direct prompts for robust translation.
- **LoRA Fine-tuning:** Efficient parameter-efficient training with high rank.
- **Optimized for Modern GPUs:** Uses bf16, TF32, and gradient checkpointing for speed and memory efficiency.

## Usage

1. **Prepare Data:**

   - Place your parallel data as `train.en.txt` and `train.vi.txt` in the same directory.
   - For testing, use `public_test.en.txt` and `public_test.vi.txt`.

2. **Install Requirements:**

   ```bash
   pip install torch transformers peft trl sacrebleu datasets
   ```

3. **Run Training:**
   ```bash
   python t2_p1.py
   ```
   Training outputs will be saved to `./vlsp_phase1_sft`.

## Notes

- Adjust hyperparameters in the `hparams` dictionary as needed for your hardware.
- The script is designed for research and competition use (e.g., VLSP medical translation).
- For best results, use a GPU with at least 24GB VRAM.

## Output

- Model checkpoints and logs in `./vlsp_phase1_sft`.
