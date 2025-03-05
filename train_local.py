import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json
from model import CAMIT_GF
from preprocess_data import preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Training (Local Version)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--num_main_layers', type=int, default=2, help='number of main transformer layers')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
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

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device - use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    model = CAMIT_GF(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_main_layers=args.num_main_layers,
        dropout=0.1  # Fixed dropout value
    ).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)  # Reduced gain for stability
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Load data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup training
    criterion = nn.MSELoss()
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
        verbose=True
    )
    
    # Enable mixed precision if CUDA is available
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    early_stopping = EarlyStopping(patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', 
                        leave=False, dynamic_ncols=True)
            
        for batch in train_pbar:
            # Move data to device
            glucose = batch['glucose'].to(device, non_blocking=True)
            carbs = batch['carbs'].to(device, non_blocking=True)
            bolus = batch['bolus'].to(device, non_blocking=True)
            basal = batch['basal'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Check for NaN inputs
            if torch.isnan(glucose).any() or torch.isnan(carbs).any() or \
               torch.isnan(bolus).any() or torch.isnan(basal).any() or \
               torch.isnan(targets).any():
                print("WARNING: NaN detected in input data, skipping batch")
                continue
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(glucose, carbs, bolus, basal)
                loss = criterion(outputs.squeeze(), targets)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print("WARNING: NaN loss detected, skipping batch")
                    continue
            
            # Mixed precision backward pass
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward pass for CPU
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', 
                      leave=False, dynamic_ncols=True)
            
        with torch.no_grad():
            for batch in val_pbar:
                glucose = batch['glucose'].to(device, non_blocking=True)
                carbs = batch['carbs'].to(device, non_blocking=True)
                bolus = batch['bolus'].to(device, non_blocking=True)
                basal = batch['basal'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                # Check for NaN inputs
                if torch.isnan(glucose).any() or torch.isnan(carbs).any() or \
                   torch.isnan(bolus).any() or torch.isnan(basal).any() or \
                   torch.isnan(targets).any():
                    print("WARNING: NaN detected in validation data, skipping batch")
                    continue
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = criterion(outputs.squeeze(), targets)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).any():
                        print("WARNING: NaN validation loss detected, skipping batch")
                        continue
                        
                val_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Close progress bars
        train_pbar.close()
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
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'best_model.pt'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
            
        # Clear cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save training history
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

if __name__ == "__main__":
    main() 