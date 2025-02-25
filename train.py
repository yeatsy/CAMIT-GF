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
    """Initialize distributed training using environment variables set by torchrun"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return int(os.environ["LOCAL_RANK"])

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')
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

def train_model(model, train_loader, val_loader, device, config, local_rank):
    """
    Distributed training loop with validation and model checkpointing
    """
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    early_stopping = EarlyStopping(patience=config['patience'])
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    if local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = torch.zeros(1).to(device)
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Train]', 
                         disable=local_rank != 0)
        
        for batch in train_pbar:
            # Move batch data to device
            glucose = batch['glucose'].to(device)
            carbs = batch['carbs'].to(device)
            bolus = batch['bolus'].to(device)
            basal = batch['basal'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(glucose, carbs, bolus, basal)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if local_rank == 0:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average loss across processes
        dist.all_reduce(train_loss)
        avg_train_loss = (train_loss / len(train_loader)).item()
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = torch.zeros(1).to(device)
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Val]',
                       disable=local_rank != 0)
        
        with torch.no_grad():
            for batch in val_pbar:
                glucose = batch['glucose'].to(device)
                carbs = batch['carbs'].to(device)
                bolus = batch['bolus'].to(device)
                basal = batch['basal'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(glucose, carbs, bolus, basal)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                if local_rank == 0:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average validation loss across processes
        dist.all_reduce(val_loss)
        avg_val_loss = (val_loss / len(val_loader)).item()
        val_losses.append(avg_val_loss)
        
        if local_rank == 0:
            logging.info(f'Epoch {epoch + 1}/{config["epochs"]} - '
                        f'Train Loss: {avg_train_loss:.4f} - '
                        f'Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': config,
                }, checkpoint_path)
                logging.info(f'Saved best model checkpoint to {checkpoint_path}')
            
            if early_stopping(avg_val_loss):
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    return train_losses, val_losses, best_val_loss

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Initialize distributed training using torchrun
    local_rank = setup_distributed()
    
    # Create config from arguments
    config = {
        'data_path': 'full_patient_dataset.csv',
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_main_layers': args.num_main_layers,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'patience': args.patience,
    }
    
    # Setup logging only on main process
    timestamp = setup_logging(local_rank)
    
    if local_rank == 0:
        logging.info("Training configuration:")
        logging.info(json.dumps(config, indent=2))
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    if local_rank == 0:
        logging.info(f"Using GPU: {torch.cuda.get_device_name(local_rank)}")
    
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = preprocess_data(config['data_path'])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,  # Reduced from 6 to match available resources
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,  # Reduced from 6 to match available resources
        pin_memory=True
    )
    
    # Initialize model
    model = CAMIT_GF(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_main_layers=config['num_main_layers']
    ).to(device)
    
    # Train model
    try:
        train_losses, val_losses, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            local_rank=local_rank
        )
        
        if local_rank == 0:
            logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    finally:
        # Clean up distributed training
        cleanup_distributed() 