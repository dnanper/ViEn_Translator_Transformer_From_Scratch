"""
Test Script to Verify Model Configuration and Data Flow

Run this BEFORE training to ensure everything is correct!
"""

import torch
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from models_best_old import BestTransformer, TransformerConfig
from utils.data_processing import DataProcessor, collate_fn
from config import Config


def test_special_tokens():
    """Test 1: Verify special token indices match"""
    print("=" * 60)
    print("TEST 1: Special Token Indices")
    print("=" * 60)
    
    # DataProcessor indices
    processor = DataProcessor(Config)
    print(f"\nDataProcessor:")
    print(f"  PAD: {processor.pad_idx}")
    print(f"  UNK: {processor.unk_idx}")
    print(f"  SOS (BOS): {processor.sos_idx}")
    print(f"  EOS: {processor.eos_idx}")
    
    # Model config indices
    config = TransformerConfig.base()
    print(f"\nTransformerConfig:")
    print(f"  PAD: {config.pad_idx}")
    print(f"  BOS: {config.bos_idx}")
    print(f"  EOS: {config.eos_idx}")
    
    # Verify match
    assert config.pad_idx == processor.pad_idx, "PAD index mismatch!"
    assert config.bos_idx == processor.sos_idx, "BOS/SOS index mismatch!"
    assert config.eos_idx == processor.eos_idx, "EOS index mismatch!"
    
    print(f"\n✅ PASS: All special token indices match!")
    return True


def test_data_shapes():
    """Test 2: Verify data shapes and masks"""
    print("\n" + "=" * 60)
    print("TEST 2: Data Shapes and Masks")
    print("=" * 60)
    
    # Create dummy batch
    batch_size = 4
    src_len = 10
    tgt_len = 12
    
    # Simulate batch from collate_fn
    src = torch.randint(4, 100, (batch_size, src_len))  # Token IDs 4-99
    tgt = torch.randint(4, 100, (batch_size, tgt_len))
    
    # Set PAD tokens
    src[:, -2:] = 0  # Last 2 tokens are PAD
    tgt[:, -3:] = 0  # Last 3 tokens are PAD
    
    # Create masks manually (như collate_fn)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    tgt_padding = (tgt != 0).unsqueeze(1).unsqueeze(3)  # [B, 1, T, 1]
    tgt_causal = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    tgt_mask = tgt_padding & tgt_causal.bool()  # [B, 1, T, T]
    
    print(f"\nShapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    print(f"  src_mask: {src_mask.shape}")
    print(f"  tgt_mask: {tgt_mask.shape}")
    
    # Verify shapes
    assert src.shape == (batch_size, src_len)
    assert tgt.shape == (batch_size, tgt_len)
    assert src_mask.shape == (batch_size, 1, 1, src_len)
    assert tgt_mask.shape == (batch_size, 1, tgt_len, tgt_len)
    
    # Test mask types
    print(f"\nMask dtypes:")
    print(f"  src_mask: {src_mask.dtype}")
    print(f"  tgt_mask: {tgt_mask.dtype}")
    
    # Verify causal mask
    assert torch.all(tgt_mask[0, 0].diagonal() == True), "Diagonal should be True"
    assert torch.all(tgt_mask[0, 0][0, 1:] == False), "Future tokens should be False"
    
    print(f"\n✅ PASS: All data shapes and masks correct!")
    return True


def test_model_forward():
    """Test 3: Model forward pass"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Forward Pass")
    print("=" * 60)
    
    # Create small model
    config = TransformerConfig.small()
    config.device = 'cpu'
    config.pad_idx = 0
    config.bos_idx = 2
    config.eos_idx = 3
    
    vocab_size = 1000
    model = BestTransformer(vocab_size, vocab_size, config)
    model.eval()
    
    print(f"\nModel created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_encoder_layers}")
    
    # Create dummy input
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(4, 100, (batch_size, src_len))
    tgt = torch.randint(4, 100, (batch_size, tgt_len))
    
    # Create masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_padding = (tgt != 0).unsqueeze(1).unsqueeze(3)
    tgt_causal = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_padding & tgt_causal.bool()
    
    print(f"\nInput shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    print(f"  src_mask: {src_mask.shape}")
    print(f"  tgt_mask: {tgt_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\nOutput shape:")
    print(f"  logits: {logits.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, tgt_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Check logits are valid
    assert not torch.isnan(logits).any(), "NaN detected in logits!"
    assert not torch.isinf(logits).any(), "Inf detected in logits!"
    
    print(f"\n✅ PASS: Model forward pass successful!")
    return True


def test_training_step():
    """Test 4: Training step (forward + loss + backward)"""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step")
    print("=" * 60)
    
    # Create model
    config = TransformerConfig.small()
    config.device = 'cpu'
    config.pad_idx = 0
    config.bos_idx = 2
    config.eos_idx = 3
    
    vocab_size = 1000
    model = BestTransformer(vocab_size, vocab_size, config)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create loss
    from models_best_old import LabelSmoothingLoss
    criterion = LabelSmoothingLoss(
        num_classes=vocab_size,
        smoothing=0.1,
        ignore_index=0  # PAD
    )
    
    # Create dummy batch
    batch_size = 4
    src_len = 12
    tgt_len = 10
    
    src = torch.randint(4, 100, (batch_size, src_len))
    tgt = torch.randint(4, 100, (batch_size, tgt_len))
    src[:, -2:] = 0  # PAD
    tgt[:, -2:] = 0  # PAD
    
    # Create masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_padding = (tgt != 0).unsqueeze(1).unsqueeze(3)
    tgt_causal = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_padding & tgt_causal.bool()
    
    # Training step
    tgt_input = tgt[:, :-1]  # Remove last token
    tgt_output = tgt[:, 1:]   # Remove first token
    tgt_mask = tgt_mask[:, :, :-1, :-1]  # Adjust mask
    
    print(f"\nTraining shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt_input: {tgt_input.shape}")
    print(f"  tgt_output: {tgt_output.shape}")
    print(f"  tgt_mask: {tgt_mask.shape}")
    
    # Forward
    logits = model(src, tgt_input, src_mask, tgt_mask)
    print(f"  logits: {logits.shape}")
    
    # Loss
    loss = criterion(logits, tgt_output)
    print(f"\nLoss: {loss.item():.4f}")
    
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0, "Loss should be positive!"
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            break
    
    assert has_grad, "No gradients computed!"
    
    # Optimizer step
    optimizer.step()
    
    print(f"\n✅ PASS: Training step successful!")
    return True


def test_beam_search():
    """Test 5: Beam search inference"""
    print("\n" + "=" * 60)
    print("TEST 5: Beam Search Inference")
    print("=" * 60)
    
    # Create model
    config = TransformerConfig.small()
    config.device = 'cpu'
    config.pad_idx = 0
    config.bos_idx = 2
    config.eos_idx = 3
    config.max_decode_length = 20
    config.beam_size = 4
    
    vocab_size = 1000
    model = BestTransformer(vocab_size, vocab_size, config)
    model.eval()
    
    # Create source sentence
    src = torch.randint(4, 100, (1, 15))  # Single sentence
    
    print(f"\nBeam search config:")
    print(f"  src: {src.shape}")
    print(f"  beam_size: {config.beam_size}")
    print(f"  max_len: {config.max_decode_length}")
    print(f"  BOS: {config.bos_idx}")
    print(f"  EOS: {config.eos_idx}")
    
    # Translate
    with torch.no_grad():
        translation = model.translate_beam(
            src,
            max_len=config.max_decode_length,
            beam_size=config.beam_size,
            length_penalty=0.6
        )
    
    print(f"\nTranslation result:")
    print(f"  Type: {type(translation)}")
    print(f"  Shape: {translation.shape if isinstance(translation, torch.Tensor) else len(translation)}")
    print(f"  First 10 tokens: {translation[:10].tolist() if isinstance(translation, torch.Tensor) else translation[:10]}")
    
    # Verify translation
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()
    
    assert len(translation) > 0, "Empty translation!"
    assert len(translation) <= config.max_decode_length, "Translation too long!"
    
    # Check if starts with BOS (if included) or just tokens
    # Check if ends with EOS
    if config.eos_idx in translation:
        print(f"  ✅ EOS token found at position {translation.index(config.eos_idx)}")
    else:
        print(f"  ⚠️ EOS token not found (may have reached max_len)")
    
    print(f"\n✅ PASS: Beam search inference successful!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING PRE-TRAINING VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Special Token Indices", test_special_tokens),
        ("Data Shapes and Masks", test_data_shapes),
        ("Model Forward Pass", test_model_forward),
        ("Training Step", test_training_step),
        ("Beam Search Inference", test_beam_search),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n❌ FAIL: {str(e)}")
            results.append((test_name, f"FAIL: {str(e)}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅" if result == "PASS" else "❌"
        print(f"{status} {test_name}: {result}")
    
    all_passed = all(result == "PASS" for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to train!")
    else:
        print("❌ SOME TESTS FAILED! Fix issues before training!")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
