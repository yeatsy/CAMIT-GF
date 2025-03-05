import torch
import torch.backends.mps
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
import json
import random
import numpy as np
from model import CAMIT_GF
from preprocess_data import preprocess_data, FORECAST_HORIZON

# Custom loss function that combines MSE with temporal smoothness and trend prediction
class GlucosePredictionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, smoothness_weight=0.1, trend_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.trend_weight = trend_weight
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, last_glucose):
        # predictions shape: [batch_size, 12] (12 future timesteps)
        # targets shape: [batch_size, 12] (12 future timesteps)
        # last_glucose shape: [batch_size, 1, 1] (last value from input sequence)
        
        # Ensure targets has the right shape
        if targets.dim() == 3:
            targets = targets.squeeze(-1)  # Remove last dimension if it exists
        
        # MSE loss with increasing weights for later predictions
        weights = torch.linspace(0.5, 1.5, predictions.size(1), device=predictions.device)
        mse_errors = self.mse(predictions, targets)  # [batch_size, 12]
        mse_per_step = mse_errors.mean(dim=0)  # [12]
        mse_loss = (weights * mse_per_step).mean()
        
        # Smoothness loss (penalize large jumps between consecutive predictions)
        diffs = predictions[:, 1:] - predictions[:, :-1]
        smoothness_loss = torch.mean(diffs ** 2)
        
        # Trend continuation loss
        last_glucose = last_glucose.squeeze(-1).squeeze(-1)  # [batch_size]
        initial_trend = predictions[:, 0] - last_glucose
        next_trend = predictions[:, 1] - predictions[:, 0]
        trend_loss = torch.mean((next_trend - initial_trend) ** 2)
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss + 
            self.smoothness_weight * smoothness_loss + 
            self.trend_weight * trend_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'smoothness': smoothness_loss.item(),
            'trend': trend_loss.item(),
            'total': total_loss.item()
        }

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Training on MPS - Using 4 hours of data to predict the next hour')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (increased for faster training)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (increased for faster training)')
    parser.add_argument('--d_model', type=int, default=64, help='model dimension (reduced for faster training)')
    parser.add_argument('--nhead', type=int, default=4, help='number of attention heads (reduced)')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='number of encoder layers (reduced)')
    parser.add_argument('--num_main_layers', type=int, default=1, help='number of main transformer layers (reduced)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--use_subset', type=float, default=0.25, help='fraction of data to use (reduced)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--eval_every', type=int, default=5, help='evaluate every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='prefetch factor for data loading')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='weight for MSE loss')
    parser.add_argument('--smoothness_weight', type=float, default=0.1, help='weight for smoothness loss')
    parser.add_argument('--trend_weight', type=float, default=0.2, help='weight for trend continuation loss')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--warmup_steps', type=int, default=200, help='warmup steps for learning rate scheduler (reduced)')
    parser.add_argument('--use_ema', action='store_true', help='use exponential moving average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='decay rate for EMA')
    parser.add_argument('--fast_mode', action='store_true', help='enable fast training mode with reduced validation')
    parser.add_argument('--val_subset', type=float, default=0.2, help='fraction of validation data to use in fast mode')
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_time_indices(batch_size, seq_len, device):
    """Generate time-of-day indices for each sequence in the batch"""
    # Randomly assign a starting time for each sequence in the batch
    # 288 = 24 hours * 12 (5-min intervals)
    start_times = torch.randint(0, 288, (batch_size,), device=device)
    
    # Create sequence of time indices for each batch item
    time_indices = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    for i, start_time in enumerate(start_times):
        # Create sequence of time indices (wrapping around at 288)
        for j in range(seq_len):
            time_indices[i, j] = (start_time + j) % 288
    
    return time_indices

# Exponential Moving Average model for better stability
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    args = parse_args()
    set_seed()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model with correct input/output dimensions
    model = CAMIT_GF(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        forecast_horizon=12  # 12 timesteps prediction
    ).to(device)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Use subset of data if specified
    if args.use_subset < 1.0:
        print(f"Using {args.use_subset*100:.1f}% of the training data")
        train_size = int(len(train_dataset) * args.use_subset)
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        train_dataset = Subset(train_dataset, train_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device.type == 'mps' else 4,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == 'mps' else 4,
        pin_memory=False
    )
    
    # Setup training
    criterion = GlucosePredictionLoss(
        mse_weight=args.mse_weight,
        smoothness_weight=args.smoothness_weight,
        trend_weight=args.trend_weight
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10)
    
    print(f"Starting training with {len(train_dataset)} training samples")
    print(f"Model parameters: d_model={args.d_model}, nhead={args.nhead}, encoder_layers={args.num_encoder_layers}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in train_pbar:
            # Get data (each sequence is 48 timesteps)
            glucose = batch['glucose'].to(device)  # [batch_size, 48, 1]
            carbs = batch['carbs'].to(device)      # [batch_size, 48, 1]
            bolus = batch['bolus'].to(device)      # [batch_size, 48, 1]
            basal = batch['basal'].to(device)      # [batch_size, 48, 1]
            targets = batch['target'].to(device)   # [batch_size, 12]
            
            # Print shapes for debugging (only first batch)
            if epoch == 0 and train_loss == 0:
                print("\nInput shapes:")
                print(f"Glucose: {glucose.shape}")
                print(f"Carbs: {carbs.shape}")
                print(f"Bolus: {bolus.shape}")
                print(f"Basal: {basal.shape}")
                print(f"Targets: {targets.shape}")
            
            # Get last glucose value for trend loss
            last_glucose = glucose[:, -1:, :]  # Keep the last dimension [batch_size, 1, 1]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(glucose, carbs, bolus, basal)  # [batch_size, 12]
            
            # Print prediction shape for debugging (only first batch)
            if epoch == 0 and train_loss == 0:
                print(f"\nPredictions shape: {predictions.shape}")
                print(f"Last glucose shape: {last_glucose.shape}\n")
            
            # Calculate loss
            loss, loss_components = criterion(predictions, targets, last_glucose)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            train_loss += loss_components['total']
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss_components['total']:.4f}",
                'mse': f"{loss_components['mse']:.4f}"
            })
        
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    glucose = batch['glucose'].to(device)
                    carbs = batch['carbs'].to(device)
                    bolus = batch['bolus'].to(device)
                    basal = batch['basal'].to(device)
                    targets = batch['target'].to(device)
                    last_glucose = glucose[:, -1:, :]  # Keep the last dimension [batch_size, 1, 1]
                    
                    predictions = model(glucose, carbs, bolus, basal)
                    loss, loss_components = criterion(predictions, targets, last_glucose)
                    val_loss += loss_components['total']
            
            val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                print(f"Saved checkpoint at epoch {epoch+1} with validation loss {val_loss:.4f}")
            
            print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f}")
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main() 