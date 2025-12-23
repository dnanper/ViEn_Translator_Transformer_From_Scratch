"""
Test DataLoader với tokenizer đã train sẵn
Chỉ cần load tokenizer và test DataLoader, không tạo data mới
"""

import os
from config import Config
from utils.data_processing import DataProcessor, get_dataloaders


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    processor = DataProcessor(Config)

    # processor.download_and_prepare_phomt()
    
    # Load tokenizer đã train
    tokenizer_dir = os.path.join(os.path.dirname(__file__), "SentencePiece-from-scratch", "tokenizer_models")
    print(f"\nLoading tokenizer from: {tokenizer_dir}")
    processor.load_tokenizer(tokenizer_dir)

    # Check data đã có chưa
    train_en = os.path.join(Config.PROCESSED_DATA_DIR, "train.en")
    if not os.path.exists(train_en):
        print(f"\nChưa có data! Chạy lệnh sau để download:")
        print(f"python -c \"from utils.data_processing import DataProcessor; from config import Config; DataProcessor(Config).download_and_prepare_phomt()\"")
        exit(1)
    
    # Prepare chỉ test dataset để test nhanh
    print("\nPreparing test dataset...")
    datasets = processor.prepare_datasets()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = get_dataloaders(datasets, processor.pad_idx)
    
    print("\n" + "-" * 60)
    for split, loader in dataloaders.items():
        print(f"{split:12s}: {len(loader):6d} batches")
    
    # Test một batch
    print("\n" + "-" * 60)
    print("Sample batch:")
    print("-" * 60)
    batch = next(iter(dataloaders['train']))
    
    print(f"src:      {batch['src'].shape}")
    print(f"tgt:      {batch['tgt'].shape}")
    print(f"src_mask: {batch['src_mask'].shape}")
    print(f"tgt_mask: {batch['tgt_mask'].shape}")
    print(batch['src'][:2,:])
    print(batch['tgt'][:2,:])
    src_text = processor.decode_sentence(batch['src'][0].tolist())
    tgt_text = processor.decode_sentence(batch['tgt'][0].tolist())
    print(f"English:    {src_text}")
    print(f"Vietnamese: {tgt_text}")

    # Test tokenization - in tokens trước khi convert sang IDs
    # print("\n" + "-" * 60)
    # print("Tokenization Test:")
    # print("-" * 60)
    
    # # Lấy câu gốc từ file
    # import pathlib
    # with open(pathlib.Path(Config.PROCESSED_DATA_DIR) / "test.en", 'r', encoding='utf-8') as f:
    #     test_en = f.readline().strip()
    # with open(pathlib.Path(Config.PROCESSED_DATA_DIR) / "test.vi", 'r', encoding='utf-8') as f:
    #     test_vi = f.readline().strip()
    
    # print(f"\nOriginal English: {test_en}")
    
    # # Tokenize và in tokens
    # tokens = processor.tokenizer.tokenize(test_en, nbest_size=1)
    # print(f"Tokens: {tokens}")
    
    # # Convert sang IDs
    # token_ids = processor.encode_sentence(test_en)
    # print(f"Token IDs: {token_ids}")
    
    # # Decode lại
    # decoded = processor.decode_sentence(token_ids)
    # print(f"Decoded: {decoded}")
    
    # print(f"\n\nOriginal Vietnamese: {test_vi}")
    # tokens = processor.tokenizer.tokenize(test_vi, nbest_size=1)
    # print(f"Tokens: {tokens}")
    # token_ids = processor.encode_sentence(test_vi)
    # print(f"Token IDs: {token_ids}")
    # decoded = processor.decode_sentence(token_ids)
    # print(f"Decoded: {decoded}")
    
    # # Decode example từ batch
    # print("\n" + "-" * 60)
    # print("Batch Example:")
    # print("-" * 60)
    
    print("\n✓ Done!")
