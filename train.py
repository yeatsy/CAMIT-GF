import torch
import torch.backends.mps  # Explicitly import MPS backend
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import os
import argparse
from datetime import datetime
import json
import numpy as np
from model import CAMIT_GF
from preprocess_data import preprocess_data, FORECAST_HORIZON

# Custom loss function that combines MSE with temporal smoothness
class GlucosePredictionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, smoothness_weight=0.1, trend_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.trend_weight = trend_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, last_glucose=None):
        # MSE loss on the predictions
        mse_loss = self.mse.forward(predictions, targets)
        
        # Smoothness loss (penalize large jumps between consecutive predictions)
        if predictions.shape[1] > 1:  # Only if we have a sequence
            diffs = predictions[:, 1:] - predictions[:, :-1]
            smoothness_loss = torch.mean(torch.square(diffs))
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # Trend continuation loss (if last_glucose is provided)
        if last_glucose is not None:
            # Calculate the trend from last observed to first prediction
            initial_trend = predictions[:, 0] - last_glucose.squeeze()
            # Calculate the trend between first and second prediction
            if predictions.shape[1] > 1:
                next_trend = predictions[:, 1] - predictions[:, 0]
                # Penalize trend reversals
                trend_loss = torch.mean(torch.square(next_trend - initial_trend))
            else:
                trend_loss = torch.tensor(0.0, device=predictions.device)
        else:
            trend_loss = torch.tensor(0.0, device=predictions.device)
        
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

def setup_distributed():
    """Initialize distributed training"""
    try:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    except (ValueError, KeyError) as e:
        print("Not running in distributed mode. Running in local mode instead.")
        return None

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--num_main_layers', type=int, default=3, help='number of main transformer layers')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--use_mps', action='store_true', help='use MPS (Metal Performance Shaders) on Apple Silicon')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='weight for MSE loss')
    parser.add_argument('--smoothness_weight', type=float, default=0.1, help='weight for smoothness loss')
    parser.add_argument('--trend_weight', type=float, default=0.2, help='weight for trend continuation loss')
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

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Try to initialize distributed training
    local_rank = setup_distributed()
    is_distributed = local_rank is not None
    
    # Set device based on mode and available hardware
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = local_rank == 0
    else:
        # Check for MPS (Apple Silicon)
        if args.use_mps:
            # More robust MPS detection
            try:
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print("Using MPS (Metal Performance Shaders) on Apple Silicon")
                else:
                    print("MPS is not available even though --use_mps was specified.")
                    print("Checking why MPS is not available:")
                    print(f"- MPS built: {torch.backends.mps.is_built()}")
                    print(f"- MPS device available: {torch.backends.mps.is_available()}")
                    device = torch.device("cpu")
            except (ImportError, AttributeError) as e:
                print(f"Error initializing MPS: {e}")
                print("Falling back to CPU.")
                device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        is_main_process = True
    
    if is_main_process:
        print(f"Using device: {device}")
    
    # Enable optimizations based on device
    if device.type == 'cuda':
        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if is_main_process and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        # MPS optimizations
        # Note: MPS backend doesn't have specific optimizations like CUDA yet
        pass
    
    # Create model
    model = CAMIT_GF(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_main_layers=args.num_main_layers,
        dropout=0.2  # Increased dropout for better regularization
    ).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)  # Reduced gain for stability
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Wrap model based on mode
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    elif device.type == 'cuda' and torch.cuda.device_count() > 1:
        if is_main_process:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Load data
    if is_main_process:
        print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create data loaders based on mode
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        # Optimize workers based on device
        num_workers = 0 if device.type == 'mps' else (4 if device.type == 'cuda' else 2)
        pin_memory = device.type == 'cuda'  # Only pin memory for CUDA
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
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
        weight_decay=args.weight_decay,
        eps=1e-8  # Increased epsilon for numerical stability
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=is_main_process
    )
    
    # Enable mixed precision only for CUDA (not supported on MPS yet)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if is_main_process:
        print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = 0.0
        train_loss_components = {'mse': 0.0, 'smoothness': 0.0, 'trend': 0.0, 'total': 0.0}
        
        # Create progress bar only on main process
        if is_main_process:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', 
                            leave=False, dynamic_ncols=True)
            train_iter = train_pbar
        else:
            train_iter = train_loader
            
        for batch in train_iter:
            # Move data to device
            glucose = batch['glucose'].to(device, non_blocking=True)
            carbs = batch['carbs'].to(device, non_blocking=True)
            bolus = batch['bolus'].to(device, non_blocking=True)
            basal = batch['basal'].to(device, non_blocking=True)
            
            # Generate target sequence for next 12 time steps
            # For now, we'll use the single target value as the final value in the sequence
            # In a real implementation, you would have all 12 future values
            targets = batch['target'].to(device, non_blocking=True)
            
            # Create a sequence of targets (placeholder - in real implementation, use actual sequence)
            # This is a simplification - ideally you would have the actual future sequence
            target_seq = torch.zeros((targets.size(0), FORECAST_HORIZON), device=device)
            target_seq[:, -1] = targets  # Set the last value to the target
            
            # Generate time-of-day indices
            time_indices = generate_time_indices(glucose.size(0), glucose.size(1), device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Check for NaN inputs
            if torch.isnan(glucose).any() or torch.isnan(carbs).any() or \
               torch.isnan(bolus).any() or torch.isnan(basal).any() or \
               torch.isnan(targets).any():
                if is_main_process:
                    print("WARNING: NaN detected in input data, skipping batch")
                continue
            
            # Get last glucose value for trend loss
            last_glucose = glucose[:, -1, :]
            
            # Forward pass with or without mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(glucose, carbs, bolus, basal, time_indices)
                    loss, loss_components = criterion(outputs, target_seq, last_glucose)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).any():
                        if is_main_process:
                            print("WARNING: NaN loss detected, skipping batch")
                        continue
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(glucose, carbs, bolus, basal, time_indices)
                loss, loss_components = criterion(outputs, target_seq, last_glucose)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    if is_main_process:
                        print("WARNING: NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            
            # Update loss tracking
            train_loss += loss_components['total']
            for k, v in loss_components.items():
                train_loss_components[k] += v
            
            # Update progress bar on main process
            if is_main_process:
                train_pbar.set_postfix({
                    'loss': f"{loss_components['total']:.4f}",
                    'mse': f"{loss_components['mse']:.4f}"
                })
        
        # Calculate average training loss
        if is_distributed:
            # Gather losses from all processes
            train_loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(train_loss_tensor)
            train_loss = train_loss_tensor.item() / len(train_loader) / dist.get_world_size()
        else:
            train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        # Create validation progress bar only on main process
        if is_main_process:
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', 
                          leave=False, dynamic_ncols=True)
            val_iter = val_pbar
        else:
            val_iter = val_loader
            
        with torch.no_grad():
            for batch in val_iter:
                glucose = batch['glucose'].to(device, non_blocking=True)
                carbs = batch['carbs'].to(device, non_blocking=True)
                bolus = batch['bolus'].to(device, non_blocking=True)
                basal = batch['basal'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                # Check for NaN inputs
                if torch.isnan(glucose).any() or torch.isnan(carbs).any() or \
                   torch.isnan(bolus).any() or torch.isnan(basal).any() or \
                   torch.isnan(targets).any():
                    if is_main_process:
                        print("WARNING: NaN detected in validation data, skipping batch")
                    continue
                
                # Generate time-of-day indices
                time_indices = generate_time_indices(glucose.size(0), glucose.size(1), device)
                
                # Create a sequence of targets (placeholder - in real implementation, use actual sequence)
                target_seq = torch.zeros((targets.size(0), FORECAST_HORIZON), device=device)
                target_seq[:, -1] = targets  # Set the last value to the target
                
                # Get last glucose value for trend loss
                last_glucose = glucose[:, -1, :]
                
                # Forward pass with mixed precision if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(glucose, carbs, bolus, basal, time_indices)
                        loss, loss_components = criterion(outputs, target_seq, last_glucose)
                        
                        # Check for NaN loss
                        if torch.isnan(loss).any():
                            if is_main_process:
                                print("WARNING: NaN loss detected, skipping batch")
                            continue
                else:
                    outputs = model(glucose, carbs, bolus, basal, time_indices)
                    loss, loss_components = criterion(outputs, target_seq, last_glucose)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).any():
                        if is_main_process:
                            print("WARNING: NaN loss detected, skipping batch")
                        continue
                        
                val_loss += loss_components['total']
                
                # Update progress bar on main process
                if is_main_process:
                    val_pbar.set_postfix({'loss': f'{loss_components["total"]:.4f}'})
        
        # Calculate average validation loss
        if is_distributed:
            # Gather losses from all processes
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss_tensor)
            val_loss = val_loss_tensor.item() / len(val_loader) / dist.get_world_size()
        else:
            val_loss = val_loss / len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Store losses
        if is_main_process:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Close progress bars
            if 'train_pbar' in locals():
                train_pbar.close()
            if 'val_pbar' in locals():
                val_pbar.close()
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'best_model.pt'))
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        # Early stopping (only check on main process but broadcast decision)
        if is_main_process and early_stopping(val_loss):
            print("Early stopping triggered")
            if is_distributed:
                # Broadcast early stopping decision to all processes
                early_stop_tensor = torch.tensor(1, device=device)
            else:
                break
        elif is_distributed:
            early_stop_tensor = torch.tensor(0, device=device)
        
        if is_distributed:
            # Broadcast early stopping decision
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item() == 1:
                break
            
        # Clear cache if CUDA is available
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save training history on main process
    if is_main_process:
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'args': vars(args)
        }
        
        with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
        print(f"Training history saved to {os.path.join(args.checkpoint_dir, 'training_history.json')}")
    
    # Clean up distributed training
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main() 