import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
from datetime import datetime
import json
from model import CAMIT_GF
from preprocess_data import preprocess_data

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

def setup_logging(log_dir="logs"):
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a timestamp for unique log file names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up logging to file
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

def train_model(model, train_loader, val_loader, device, config):
    """
    Training loop with validation and model checkpointing
    """
    # Move model to device and set to train mode
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Create directory for model checkpoints
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Train]')
        
        for batch in train_pbar:
            # Move all batch data to device
            glucose = batch['glucose'].to(device)
            carbs = batch['carbs'].to(device)
            bolus = batch['bolus'].to(device)
            basal = batch['basal'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(glucose, carbs, bolus, basal)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                # Move all batch data to device
                glucose = batch['glucose'].to(device)
                carbs = batch['carbs'].to(device)
                bolus = batch['bolus'].to(device)
                basal = batch['basal'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(glucose, carbs, bolus, basal)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log the losses
        logging.info(f'Epoch {epoch + 1}/{config["epochs"]} - '
                    f'Train Loss: {avg_train_loss:.4f} - '
                    f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,  # Save config for model reconstruction
            }, checkpoint_path)
            logging.info(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return train_losses, val_losses, best_val_loss

# Set up CUDA device
def setup_device():
    if torch.cuda.is_available():
        # In SLURM environment, we should automatically get the correct GPU
        device = torch.device("cuda")
        # Print GPU info for logging
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Using GPU: {gpu_name}")
        # Set default tensor type to cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device")
        return device
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
        return device

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': 'full_patient_dataset.csv',
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 2,
        'num_main_layers': 2,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 7,
    }
    
    # Setup logging
    timestamp = setup_logging()
    
    # Log configuration
    logging.info("Training configuration:")
    logging.info(json.dumps(config, indent=2))
    
    # Set device and enable CUDA optimizations
    device = setup_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logging.info("CUDA optimization enabled")
    
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = preprocess_data(config['data_path'])
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=4 if device.type == 'cuda' else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Initialize model
    model = CAMIT_GF(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_main_layers=config['num_main_layers']
    )
    
    # Train model
    train_losses, val_losses, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}") 