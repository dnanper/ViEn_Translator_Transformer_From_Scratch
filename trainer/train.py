"""
Training Script for Vietnamese-English Translation

Complete pipeline:
1. Load tokenized data
2. Create model
3. Train with warmup + lr decay
4. Validate and save checkpoints
5. Track metrics and visualize
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models_best import BestTransformer, TransformerConfig, LabelSmoothingLoss
from utils.data_processing import DataProcessor, get_dataloaders
from config import Config


class Trainer:
    """
    Trainer for Neural Machine Translation
    """
    
    def __init__(self, model: BestTransformer, config: TransformerConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 save_dir: str):
        """
        Args:
            model: Transformer model
            config: Model configuration
            train_loader: Training dataloader
            val_loader: Validation dataloader
            save_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            num_classes=model.tgt_vocab_size,
            smoothing=config.label_smoothing,
            ignore_index=config.pad_idx
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / self.config.warmup_steps
            else:
                # Inverse square root decay
                return (self.config.warmup_steps / step) ** 0.5
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.config.device)  # [batch_size, src_len]
            tgt = batch['tgt'].to(self.config.device)  # [batch_size, tgt_len]
            src_mask = batch['src_mask'].to(self.config.device)  # [batch_size, 1, 1, src_len]
            tgt_mask = batch['tgt_mask'].to(self.config.device)  # [batch_size, 1, tgt_len, tgt_len]
            
            # Validate token IDs (防止 index out of bounds)
            vocab_size = self.model.tgt_vocab_size
            src = torch.clamp(src, 0, vocab_size - 1)
            tgt = torch.clamp(tgt, 0, vocab_size - 1)
            
            # Prepare input and output
            tgt_input = tgt[:, :-1]   # Remove last token (for input)
            tgt_output = tgt[:, 1:]   # Remove first token (for output)
            tgt_mask = tgt_mask[:, :, :-1, :-1]  # Adjust mask for tgt_input
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)  # [batch_size, tgt_len-1, vocab_size]
            
            # Compute loss
            loss = self.criterion(logits, tgt_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log every 100 steps
            if self.global_step % 100 == 0:
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            src_mask = batch['src_mask'].to(self.config.device)
            tgt_mask = batch['tgt_mask'].to(self.config.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = self.criterion(logits, tgt_output)
            total_loss += loss.item()
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int, save_every: int = 1, resume_from: Optional[str] = None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print("=" * 60)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("=" * 60)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {self.epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"  Time:       {epoch_time:.1f}s")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pt')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}\n")
            
            # Save latest checkpoint (for resuming)
            self.save_checkpoint('latest.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        # Plot training curves
        self.plot_training_curves()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print("=" * 60)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        steps = range(len(self.history['learning_rate']))
        ax2.plot(steps, self.history['learning_rate'])
        ax2.set_xlabel('Step (x100)')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        print(f"✓ Training curves saved to {self.save_dir / 'training_curves.png'}")


def main():
    """Main training function"""
    
    # ========== Configuration ==========
    
    # Get project root directory (parent of trainer/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Paths (relative to project root)
    SAVE_DIR = PROJECT_ROOT / "checkpoints" / "best_model_en2vi"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    
    # Model configuration - THAY ĐỔI KÍCH CỠ MODEL TẠI ĐÂY:
    # .small() - 256d, 4 layers, ~60M params (fast training)
    # .base()  - 512d, 6 layers, ~65M params (balanced) ⭐ RECOMMENDED
    # .large() - 1024d, 6 layers, ~213M params (best quality)
    # .deep()  - 512d, 12 layers (very deep)
    config = TransformerConfig.base()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.max_len = Config.MAX_LEN
    config.dropout = 0.1
    config.label_smoothing = 0.1
    config.warmup_steps = 8000
    config.learning_rate = 1e-4
    config.grad_clip = 1.0
    
    # Training configuration
    BATCH_SIZE = Config.BATCH_SIZE  # Adjust based on GPU memory
    NUM_EPOCHS = 30
    
    print("=" * 60)
    print("English-Vietnamese Translation Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {config.d_model}d, {config.n_encoder_layers} layers")
    print(f"  Direction: English → Vietnamese")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {config.max_len}")
    print(f"  Device: {config.device}")
    print("=" * 60)
    
    # ========== Initialize Data Processor ==========
    
    print("\nInitializing data processor...")
    processor = DataProcessor(Config)
    processor.load_tokenizer(TOKENIZER_DIR)
    
    VOCAB_SIZE = processor.vocab_size
    print(f"  Vocabulary size: {VOCAB_SIZE:,}")
    
    # ========== Create DataLoaders ==========
    
    print("\nPreparing datasets...")
    datasets = processor.prepare_datasets()  # Returns {'train', 'validation', 'test'}
    dataloaders = get_dataloaders(datasets, processor.pad_idx, batch_size=BATCH_SIZE)
    
    # Use proper train/validation/test splits (no data leakage)
    train_loader = dataloaders['train']
    val_loader = dataloaders['validation']
    test_loader = dataloaders['test']
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(datasets['train']):,}")
    print(f"  Validation samples: {len(datasets['validation']):,}")
    print(f"  Test samples: {len(datasets['test']):,}")
    print(f"  Training batches: {len(train_loader):,}")
    
    # ========== Create Model ==========
    
    print("\nCreating model...")
    # Update config with correct pad_idx from processor
    config.pad_idx = processor.pad_idx
    config.bos_idx = processor.sos_idx  # Use SOS as BOS
    config.eos_idx = processor.eos_idx
    
    model = BestTransformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        config=config
    )
    
    # ========== Create Trainer ==========
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=SAVE_DIR
    )
    
    # ========== Train ==========
    
    # To resume from checkpoint, uncomment:
    # trainer.train(NUM_EPOCHS, save_every=1, resume_from=os.path.join(SAVE_DIR, 'latest.pt'))
    
    trainer.train(NUM_EPOCHS, save_every=1)
    
    print("\n✓ Training complete!")
    print(f"  Best model saved to: {SAVE_DIR}/best_model.pt")
    print(f"  Training curves: {SAVE_DIR}/training_curves.png")


if __name__ == '__main__':
    # Import collate_fn here to avoid circular import
    from utils.data_processing import collate_fn
    main()
