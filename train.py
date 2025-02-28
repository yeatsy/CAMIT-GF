import torch
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
from model import CAMIT_GF
from preprocess_data import preprocess_data

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return int(os.environ["LOCAL_RANK"])

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--num_main_layers', type=int, default=2, help='number of main transformer layers')
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def setup_logging(local_rank, log_dir="logs"):
    if local_rank == 0:  # Only create logs on main process
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return timestamp
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def main():
    args = parse_args()
    local_rank = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model
    model = CAMIT_GF(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_main_layers=args.num_main_layers
    ).to(device)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Load data
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create data loaders
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
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        train_loss = torch.zeros(1).to(device)
        
        # Create progress bars only on rank 0
        if local_rank == 0:
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
            targets = batch['target'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(glucose, carbs, bolus, basal)
                loss = criterion(outputs.squeeze(), targets)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.detach()
            
            # Update progress bar on rank 0
            if local_rank == 0:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average loss across processes
        dist.all_reduce(train_loss)
        train_loss = train_loss.item() / len(train_loader) / dist.get_world_size()
        
        # Validation
        model.eval()
        val_loss = torch.zeros(1).to(device)
        
        # Create validation progress bar only on rank 0
        if local_rank == 0:
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
                
                with torch.cuda.amp.autocast():
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.detach()
                
                # Update progress bar on rank 0
                if local_rank == 0:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average validation loss across processes
        dist.all_reduce(val_loss)
        val_loss = val_loss.item() / len(val_loader) / dist.get_world_size()
        
        # Print progress on main process
        if local_rank == 0:
            # Close progress bars
            train_pbar.close()
            val_pbar.close()
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        
        # Clear cache
        torch.cuda.empty_cache()
    
    cleanup_distributed()

if __name__ == "__main__":
    main() 