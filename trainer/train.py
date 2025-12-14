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

from models_best import BestTransformer, TransformerConfig
from utils.data_processing import DataProcessor, get_dataloaders
from config import Config


class Trainer:
    """
    Trainer for Neural Machine Translation
    """
    
    def __init__(self, model: BestTransformer, config: TransformerConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 save_dir: str, processor=None):
        """
        Args:
            model: Transformer model
            config: Model configuration
            train_loader: Training dataloader
            val_loader: Validation dataloader
            save_dir: Directory to save checkpoints and logs
            processor: DataProcessor for decoding (optional, needed for BLEU)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.processor = processor
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function - Using PyTorch's CrossEntropyLoss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.pad_idx,
            label_smoothing=config.label_smoothing,
            reduction='mean'
        )
        print(f"✓ Using CrossEntropyLoss with label_smoothing={config.label_smoothing}")
        
        # Optimizer - Use AdamW (better weight decay than Adam)
        # Try fused version for 30-50% faster training if available
        try:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                fused=True  # PyTorch 2.0+ - faster CUDA kernel
            )
            print("✓ Using fused AdamW optimizer (faster)")
        except:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay
            )
            print("✓ Using standard AdamW optimizer")
        
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
            'learning_rate': [],
            'bleu_scores': []  # Track BLEU each epoch
        }
    
    def _create_scheduler(self):
        """Create learning rate scheduler: Warmup + Hold + Cosine Decay
        
        Strategy for large dataset (2.4M samples):
        1. Warmup: 0 → peak LR (fast initial learning)
        2. Hold: Keep peak LR (allow model to optimize properly)
        3. Decay: Slow cosine decay (fine-tuning)
        """
        import math
        
        # Calculate steps
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * 15  # 15 epochs total
        
        # Phase 1: Warmup (0 → peak)
        warmup_steps = self.config.warmup_steps
        
        # Phase 2: Hold at peak LR (brief stabilization after warmup)
        # Keep peak LR for 1 epoch to allow proper optimization
        hold_steps = steps_per_epoch * 1  # Hold for 1 epoch
        
        # Phase 3: Cosine decay
        decay_start = warmup_steps + hold_steps
        
        def lr_lambda(step):        
            if step < warmup_steps:
                # Phase 1: Linear warmup
                return step / warmup_steps
            
            elif step < decay_start:
                # Phase 2: HOLD at peak LR (return 1.0)
                return 1.0
            
            else:
                # Phase 3: Cosine decay
                progress = (step - decay_start) / (total_steps - decay_start)
                progress = min(1.0, progress)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                min_lr_ratio = 0.1
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        print(f"\nScheduler Info:")
        print(f"  Steps/epoch: {steps_per_epoch:,}")
        print(f"  Warmup: {warmup_steps:,} steps ({warmup_steps/steps_per_epoch:.2f} epochs)")
        print(f"  Hold: {hold_steps:,} steps (1 epoch)")
        print(f"  Decay starts at step {decay_start:,} (~epoch {decay_start/steps_per_epoch:.1f})")
        print(f"  Cosine decay: {total_steps - decay_start:,} steps (~{(total_steps-decay_start)/steps_per_epoch:.1f} epochs)")
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    

    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            src_mask = batch['src_mask'].to(self.config.device)
            tgt_mask = batch['tgt_mask'].to(self.config.device)
            
            # Prepare input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            # CrossEntropyLoss expects: (N, C, ...) for input and (N, ...) for target
            # Reshape: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            tgt_flat = tgt_output.reshape(-1)
            loss = self.criterion(logits_flat, tgt_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
            
            # Track gradient norms occasionally (every 100 steps for monitoring)
            if self.global_step % 100 == 0:
                if not hasattr(self, 'grad_norms'):
                    self.grad_norms = []
                self.grad_norms.append(grad_norm.item())
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            self.global_step += 1
            
            # Log every 100 steps
            if self.global_step % 100 == 0:
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Track metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{grad_norm.item():.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
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
            # CrossEntropyLoss expects: (N, C, ...) for input and (N, ...) for target
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            tgt_flat = tgt_output.reshape(-1)
            loss = self.criterion(logits_flat, tgt_flat)
            total_loss += loss.item()
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def compute_bleu(self, num_samples: int = 500) -> float:
        """
        Compute BLEU score on validation set
        
        Args:
            num_samples: Number of samples to evaluate (default 500 for speed)
        
        Returns:
            BLEU-4 score
        """
        from evaluate import compute_bleu_with_tokens
        
        self.model.eval()
        references = []
        hypotheses = []
        
        print(f"\nComputing BLEU on {num_samples} validation samples...")
        
        for i, batch in enumerate(self.val_loader):
            if i * batch['src'].size(0) >= num_samples:
                break
            
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            src_mask = batch['src_mask'].to(self.config.device)
            
            # Greedy decode
            batch_size = src.size(0)
            max_len = self.config.max_decode_length
            
            # Start with SOS token
            decoder_input = torch.full((batch_size, 1), self.config.bos_idx, 
                                     dtype=torch.long, device=self.config.device)
            
            for _ in range(max_len):
                # Create causal mask for decoder
                tgt_len = decoder_input.size(1)
                tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.config.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).bool()
                
                # Forward pass
                logits = self.model(src, decoder_input, src_mask, tgt_mask)
                
                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == self.config.eos_idx).all():
                    break
            
            # Collect references and hypotheses
            for j in range(batch_size):
                ref = tgt[j].cpu().tolist()
                hyp = decoder_input[j].cpu().tolist()
                
                references.append(ref)
                hypotheses.append(hyp)
        
        # Compute BLEU using evaluate module
        if self.processor is None:
            print("⚠️  Processor not available, skipping BLEU calculation")
            return 0.0
        
        results = compute_bleu_with_tokens(references, hypotheses, self.processor, max_n=4)
        bleu_4 = results.get('bleu-4', 0.0)
        
        print(f"✓ BLEU-4: {bleu_4:.2f}")
        return bleu_4
    
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
            
            # Compute BLEU every epoch (or every N epochs for speed)
            # Disabled for faster training - enable after training stabilizes
            if False and self.epoch % 5 == 0:  # Disabled
                bleu_score = self.compute_bleu(num_samples=200)
                self.history['bleu_scores'].append(bleu_score)
            else:
                self.history['bleu_scores'].append(None)
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Average gradient norm for monitoring
            if hasattr(self, 'grad_norms') and len(self.grad_norms) > 0:
                avg_grad_norm = sum(self.grad_norms) / len(self.grad_norms)
            else:
                avg_grad_norm = 0
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {self.epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            if self.history['bleu_scores'][-1] is not None:
                print(f"  BLEU-4:     {self.history['bleu_scores'][-1]:.2f}")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"  Avg Grad:   {avg_grad_norm:.2f}")
            print(f"  Time:       {epoch_time:.1f}s")
            print(f"{'='*60}\n")
            
            # Check for training issues
            if avg_grad_norm < 0.01:
                print("⚠️  WARNING: Average gradient is very small! Model may not be learning.")
            if abs(train_loss - val_loss) < 0.001:
                print("⚠️  WARNING: Train and Val loss are identical! This is unusual.")
            
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # BLEU scores
        ax2 = axes[0, 1]
        bleu_epochs = [i for i, score in enumerate(self.history['bleu_scores'], 1) if score is not None]
        bleu_values = [score for score in self.history['bleu_scores'] if score is not None]
        if bleu_values:
            ax2.plot(bleu_epochs, bleu_values, label='BLEU-4', marker='D', 
                    color='green', linewidth=2, markersize=8)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('BLEU-4 Score', fontsize=12)
            ax2.set_title('BLEU Score Progress', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3 = axes[1, 0]
        steps = range(len(self.history['learning_rate']))
        ax3.plot(steps, self.history['learning_rate'], color='orange', linewidth=2)
        ax3.set_xlabel('Step (x100)', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Gradient norms
        ax4 = axes[1, 1]
        if hasattr(self, 'grad_norms') and len(self.grad_norms) > 0:
            ax4.plot(self.grad_norms, color='red', linewidth=1.5, alpha=0.7)
            ax4.axhline(y=self.config.grad_clip, color='black', linestyle='--', 
                       label=f'Clip threshold ({self.config.grad_clip})')
            ax4.set_xlabel('Step (x100)', fontsize=12)
            ax4.set_ylabel('Gradient Norm', fontsize=12)
            ax4.set_title('Gradient Norms', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
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
    
    # CRITICAL: Higher regularization for large dataset (2.4M samples)
    config.dropout = 0.1              # Increased from 0.1 to reduce overfitting
    config.attention_dropout = 0.1    # Add attention dropout
    config.label_smoothing = 0.05
    
    # CRITICAL: Warmup + HOLD + Decay strategy
    config.warmup_steps = 10000       # Fast warmup (~26% of epoch 1)
    config.learning_rate = 1.5e-4     # Moderate LR, will hold at peak for 5 epochs
    config.grad_clip = 5.0            # Increased from 1.0 - allow larger gradients
    
    # Training configuration
    BATCH_SIZE = 48
    NUM_EPOCHS = 15
    
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
        processor=processor,
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
