import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from models_best import BestTransformer
from utils.data_processing import DataProcessor
from config import Config

def translate_sentence(text: str):
    PROJECT_ROOT = Path(__file__).parent
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    DEVICE = 'cpu'  # Force CPU for inference
    
    print(f"Device: {DEVICE}")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    config.device = DEVICE
    
    # Load tokenizer
    processor = DataProcessor(Config)
    processor.load_tokenizer(str(TOKENIZER_DIR))
    
    # Create model
    model = BestTransformer(
        src_vocab_size=processor.vocab_size,
        tgt_vocab_size=processor.vocab_size,
        config=config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Tokenize
    src_ids = processor.encode_sentence(text, add_sos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
    
    # Translate (greedy - nhanh nhất)
    with torch.no_grad():
        tgt_ids = model.translate_greedy(src_tensor)
    
    # Decode
    translation = processor.decode_sentence(tgt_ids, skip_special_tokens=True)
    
    return translation

if __name__ == '__main__':
    text = "This exam is so hard because I didn't study and sleep all day."
    print(f"\nInput:  {text}")
    result = translate_sentence(text)
    print(f"Output: {result}")