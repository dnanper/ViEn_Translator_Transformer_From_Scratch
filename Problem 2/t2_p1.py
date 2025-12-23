import os
import torch
import random
import gc
import sacrebleu
from datasets import IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. CRITICAL STABILITY SETTINGS
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevents deadlocks
torch.backends.cuda.matmul.allow_tf32 = True    # Boosts speed on RTX 5090
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# 0. CONFIGURATION (RTX 5090 + 3B MODEL OPTIMIZED)
# ==============================================================================
hparams = {
    # UPGRADE: Using 3B instead of 1.5B because your 5090 (32GB) can handle it.
    # 3B is significantly smarter for medical concepts.
    "model_name": "Qwen/Qwen3-1.7B",

    "train_en": "train.en.txt",
    "train_vi": "train.vi.txt",
    "test_en": "public_test.en.txt",
    "test_vi": "public_test.vi.txt",

    # UPGRADE: Native BF16 is faster & more accurate than 4-bit on RTX 5090
    "dtype": torch.bfloat16,

    # --- LoRA (High Rank for Full-Parameter Behavior) ---
    "lora": {
        "r": 64,              # Rank 64 mimics full-finetuning capacity
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    },

    # --- Training ---
    "training": {
        "output_dir": "./vlsp_phase1_sft",
        # RAM SAFETY: Keep workers at 0 to prevent std::bad_alloc
        "dataloader_num_workers": 116, 
        "per_device_train_batch_size": 4,   
        "gradient_accumulation_steps": 8,   
        "gradient_checkpointing": True,     # Saves VRAM
        "learning_rate": 2e-4, 
        "max_steps": 30000,                 # Streaming uses steps, not epochs (approx 1 epoch)
        "bf16": True,
        "tf32": True,
        "logging_steps": 200,
        "save_steps": 1000,
        "report_to": "none",
    }
}

# ==============================================================================
# 1. STREAMING DATA LOADER (SOLVES std::bad_alloc)
# ==============================================================================
def data_generator(en_path, vi_path):
    """
    Reads file line-by-line. Uses 0GB RAM.
    Implements Bosch@AI Bidirectional Strategy + Dynamic Prompting.
    """
    system_prompt = (
        "<|im_start|>system\n"
        "You are a professional medical translator specializing in clinical and research texts. "
        "Translate accurately, preserving technical terms, formal tone, and medical context. "
        "Avoid hallucinations; ensure translations are ethically sound and suitable for healthcare use."
        "<|im_end|>\n"
    )
    
    # Infinite loop logic for IterableDataset
    while True:
        with open(en_path, 'r', encoding='utf-8') as f_en, \
             open(vi_path, 'r', encoding='utf-8') as f_vi:
            
            for en, vi in zip(f_en, f_vi):
                en, vi = en.strip(), vi.strip()
                if not en or not vi: continue
                
                # [cite_start]Bosch Strategy: Train both directions [cite: 78]
                directions = [("English", "Vietnamese", en, vi), 
                              ("Vietnamese", "English", vi, en)]
                
                for src_lang, tgt_lang, src_text, tgt_text in directions:
                    # [cite_start]Dynamic Prompting [cite: 8]
                    if random.random() < 0.6: # 60% Instructional
                        prompt = (f"{system_prompt}<|im_start|>user\nTranslate the following medical text from {src_lang} to {tgt_lang}.\n"
                                  f"Source Text: {src_text}\n<|im_end|>\n<|im_start|>assistant\n{tgt_text}<|im_end|>")
                    else: # 40% Direct
                        prompt = (f"{system_prompt}<|im_start|>user\n{src_text}\n({src_lang} -> {tgt_lang})<|im_end|>\n"
                                  f"<|im_start|>assistant\n{tgt_text}<|im_end|>")
                    
                    yield {"text": prompt}

# Initialize Streaming Dataset
print("Initializing Zero-RAM Streaming Dataset...")
train_dataset = IterableDataset.from_generator(
    data_generator, 
    gen_kwargs={"en_path": hparams["train_en"], "vi_path": hparams["train_vi"]}
)

# Static Validation Subset (Safe for RAM as it's small)
def load_val_subset(en_path, vi_path, n=50):
    data = []
    with open(en_path) as f_en, open(vi_path) as f_vi:
        for i, (en, vi) in enumerate(zip(f_en, f_vi)):
            if i >= n: break
            prompt = f"<|im_start|>system\nYou are a professional medical translator.<|im_end|>\n<|im_start|>user\nTranslate the following medical text from English to Vietnamese:\n{en.strip()}<|im_end|>\n<|im_start|>assistant\n"
            data.append({"text": prompt, "ref": vi.strip()})
    return data

val_subset = load_val_subset(hparams["test_en"], hparams["test_vi"])

# ==============================================================================
# 2. MODEL SETUP (NATIVE BF16)
# ==============================================================================
print(f"Loading {hparams['model_name']}...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./vlsp_phase1_sft/final_adapter")
model.config.use_cache = False 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "./vlsp_phase1_sft/final_adapter",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # BẮT BUỘC để match Phase 1
tokenizer.padding_side = "right" # Training side

peft_config = LoraConfig(**hparams["lora"])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==============================================================================
# 3. CALLBACKS & EXECUTION
# ==============================================================================
class BLEUCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_data):
        self.tokenizer = tokenizer
        self.eval_data = eval_data
    
    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate every 200 steps
        if state.global_step % 1000 == 0 and state.global_step > 0:
            print(f"\n[Step {state.global_step}] Interval Check...")
            model.eval()
            refs, preds = [], []
            self.tokenizer.padding_side = "left" # Inference side
            
            for item in self.eval_data:
                inputs = self.tokenizer(item["text"], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
                pred = self.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
                preds.append(pred); refs.append(item["ref"])
            
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            print(f">>> Phase 1 BLEU: {bleu.score:.2f}")
            
            self.tokenizer.padding_side = "right"
            model.train()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(**hparams["training"]),
    callbacks=[BLEUCallback(tokenizer, val_subset)],
)

print("Starting Phase 1 SFT...")
trainer.train()

print("Saving Final Adapter...")
final_path = os.path.join(hparams["training"]["output_dir"], "final_adapter")
trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Done! Saved to {final_path} [cite: 254, 256]")