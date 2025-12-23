"""
Configuration file for Vietnamese-English Neural Machine Translation
"""
import os
from pathlib import Path

# Get the absolute path of the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

class Config:
    # Data settings
    DATA_DIR = str(PROJECT_ROOT / "data")
    RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
    PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")
    DATASET_NAME = "ura-hcmut/PhoMT"  # PhoMT dataset with ~2.9M parallel sentences
    
    # Model hyperparameters
    D_MODEL = 512              # Embedding dimension
    N_HEADS = 8                # Number of attention heads
    N_ENCODER_LAYERS = 6       # Number of encoder layers
    N_DECODER_LAYERS = 6       # Number of decoder layers
    D_FF = 2048                # Feed-forward dimension
    DROPOUT = 0.1              # Dropout rate
    
    # Vocabulary settings
    VOCAB_SIZE = 32000         # Shared vocabulary size
    MAX_LEN = 256              # Maximum sequence length
    
    # Training settings
    BATCH_SIZE = 32            # Batch size (will be adjusted for dynamic batching)
    MAX_TOKENS_PER_BATCH = 4096  # For dynamic batching
    NUM_EPOCHS = 30            # Number of training epochs
    LEARNING_RATE = 0.0001     # Base learning rate (will be adjusted by scheduler)
    WARMUP_STEPS = 4000        # Learning rate warmup steps
    LABEL_SMOOTHING = 0.1      # Label smoothing epsilon
    WEIGHT_DECAY = 0.01        # AdamW weight decay
    GRAD_CLIP = 1.0            # Gradient clipping norm
    
    # Optimizer settings
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.98
    ADAM_EPSILON = 1e-9
    
    # Beam search settings
    BEAM_WIDTH = 5
    LENGTH_PENALTY = 0.6
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    # Checkpoint settings
    CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")
    SAVE_EVERY = 1             # Save checkpoint every N epochs
    
    # Evaluation settings
    EVAL_EVERY = 1000          # Evaluate every N steps
    SAVE_TRANSLATIONS = True   # Save sample translations
    
    # Device settings
    USE_CUDA = True
    MIXED_PRECISION = True     # Use FP16 training
    
    # Gemini API settings (optional)
    GEMINI_API_KEY = None      # Set this to use Gemini Score
    USE_GEMINI_SCORE = False   # Enable Gemini Score evaluation
    
    # Advanced features
    USE_WEIGHT_TYING = True    # Tie embedding and output weights
    USE_PRE_LN = False         # Use Pre-LN instead of Post-LN
    USE_SWIGLU = False         # Use SwiGLU activation (set to True for better performance)
    USE_CURRICULUM_LEARNING = False  # Sort data by length
    USE_BACK_TRANSLATION = False     # Augment with back-translated data
    
    @classmethod
    def get_device(cls):
        import torch
        if cls.USE_CUDA and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
