import torch
import torch.nn as nn
import math
import argparse
import os
import json

# Positional Encoding module for transformer inputs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# Custom transformer layer with self-attention on glucose and cross-attention to carbs, bolus, and basal
class MultiCrossAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, layer_scale_init=1e-4):
        super().__init__()
        # Attention layers with batch_first=True for MPS compatibility
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_carbs = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_bolus = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_basal = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # Replace ReLU with GELU
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        # Layer normalization for each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        
        # Layer scale parameters
        self.ls1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls3 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls4 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls5 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, glucose_hidden, carbs_hidden, bolus_hidden, basal_hidden):
        # Self-attention on glucose sequence with layer scale
        attn_output, _ = self.self_attn(glucose_hidden, glucose_hidden, glucose_hidden)
        glucose_hidden = glucose_hidden + self.dropout(self.ls1.unsqueeze(0).unsqueeze(0) * attn_output)
        glucose_hidden = self.norm1(glucose_hidden)

        # Cross-attention to carbs sequence
        attn_output, _ = self.cross_attn_carbs(glucose_hidden, carbs_hidden, carbs_hidden)
        glucose_hidden = glucose_hidden + self.dropout(self.ls2.unsqueeze(0).unsqueeze(0) * attn_output)
        glucose_hidden = self.norm2(glucose_hidden)

        # Cross-attention to bolus sequence
        attn_output, _ = self.cross_attn_bolus(glucose_hidden, bolus_hidden, bolus_hidden)
        glucose_hidden = glucose_hidden + self.dropout(self.ls3.unsqueeze(0).unsqueeze(0) * attn_output)
        glucose_hidden = self.norm3(glucose_hidden)

        # Cross-attention to basal sequence
        attn_output, _ = self.cross_attn_basal(glucose_hidden, basal_hidden, basal_hidden)
        glucose_hidden = glucose_hidden + self.dropout(self.ls4.unsqueeze(0).unsqueeze(0) * attn_output)
        glucose_hidden = self.norm4(glucose_hidden)

        # Feedforward network
        ff_output = self.feed_forward(glucose_hidden)
        glucose_hidden = glucose_hidden + self.dropout(self.ls5.unsqueeze(0).unsqueeze(0) * ff_output)
        glucose_hidden = self.norm5(glucose_hidden)

        return glucose_hidden

# Main CAMIT-GF model class
class CAMIT_GF(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_main_layers, dropout=0.1):
        super().__init__()
        # Input projection layers for each input type
        self.input_proj_G = nn.Linear(1, d_model)  # Glucose
        self.input_proj_C = nn.Linear(1, d_model)  # Carbs
        self.input_proj_B = nn.Linear(1, d_model)  # Bolus insulin
        self.input_proj_A = nn.Linear(1, d_model)  # Basal insulin
        # Positional encoding shared across inputs
        self.pos_encoding = PositionalEncoding(d_model)
        # Transformer encoders for each input type, with batch_first=True
        self.encoder_G = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.encoder_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.encoder_B = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.encoder_A = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True),
            num_layers=num_encoder_layers
        )
        # Main transformer with multiple custom layers
        self.main_transformer = nn.ModuleList([
            MultiCrossAttentionTransformerLayer(d_model, nhead, dropout) for _ in range(num_main_layers)
        ])
        # Prediction head to output a single glucose value
        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, glucose, carbs, bolus, basal):
        # Project inputs to d_model dimensions
        glucose_proj = self.input_proj_G(glucose)
        carbs_proj = self.input_proj_C(carbs)
        bolus_proj = self.input_proj_B(bolus)
        basal_proj = self.input_proj_A(basal)

        # Add positional encoding
        glucose_proj = self.pos_encoding(glucose_proj)
        carbs_proj = self.pos_encoding(carbs_proj)
        bolus_proj = self.pos_encoding(bolus_proj)
        basal_proj = self.pos_encoding(basal_proj)

        # Encode each input using respective encoders
        glucose_hidden = self.encoder_G(glucose_proj)
        carbs_hidden = self.encoder_C(carbs_proj)
        bolus_hidden = self.encoder_B(bolus_proj)
        basal_hidden = self.encoder_A(basal_proj)

        # Process through main transformer layers
        for layer in self.main_transformer:
            glucose_hidden = layer(glucose_hidden, carbs_hidden, bolus_hidden, basal_hidden)

        # Extract last hidden state for prediction
        last_hidden = glucose_hidden[:, -1, :]
        prediction = self.prediction_head(last_hidden)
        return prediction

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CAMIT-GF Training')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training')
    args = parser.parse_args()

    # Set up CUDA and optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Check GPU configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load and preprocess actual data
    from preprocess_data import preprocess_data
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create data loaders with GPU optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Enable automatic mixed precision if requested
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # Hyperparameter search space
    param_grid = {
        'd_model': [32, 64, 128],
        'nhead': [4, 8],
        'num_encoder_layers': [1, 2, 3],
        'num_main_layers': [1, 2, 3],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.0001, 0.001],
        'weight_decay': [0.01, 0.001]
    }

    # Function to train model for a few epochs to evaluate hyperparameters
    def evaluate_hyperparameters(params, num_search_epochs=5):
        model = CAMIT_GF(
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            num_main_layers=params['num_main_layers'],
            dropout=params['dropout']
        ).to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        loss_fn = nn.MSELoss()
        best_val_loss = float('inf')
        
        for epoch in range(num_search_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                # Move batch to device
                glucose = batch['glucose'].to(device, non_blocking=True)
                carbs = batch['carbs'].to(device, non_blocking=True)
                bolus = batch['bolus'].to(device, non_blocking=True)
                basal = batch['basal'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = loss_fn(outputs.squeeze(), targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    glucose = batch['glucose'].to(device, non_blocking=True)
                    carbs = batch['carbs'].to(device, non_blocking=True)
                    bolus = batch['bolus'].to(device, non_blocking=True)
                    basal = batch['basal'].to(device, non_blocking=True)
                    targets = batch['target'].to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(glucose, carbs, bolus, basal)
                        loss = loss_fn(outputs.squeeze(), targets)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            
            print(f"Epoch {epoch+1}/{num_search_epochs} - Val Loss: {val_loss:.4f}")
            
            # Clear cache after each epoch
            torch.cuda.empty_cache()
        
        return best_val_loss

    # Rest of the hyperparameter search code...
    print("Starting hyperparameter search...")
    best_params = None
    best_val_loss = float('inf')
    
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    total_combinations = len(param_combinations)
    
    print(f"Testing {total_combinations} hyperparameter combinations...")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {i}/{total_combinations}")
        print("Parameters:", params)
        
        val_loss = evaluate_hyperparameters(params)
        print(f"Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            print("New best configuration found!")
            
            # Save best parameters so far
            with open(os.path.join(args.checkpoint_dir, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_params, f, indent=2)

    print("\nHyperparameter search completed.")
    print("Best parameters:", best_params)
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Final training with best parameters
    print("\nStarting final training with best parameters...")
    
    # Initialize model with best parameters
    model = CAMIT_GF(
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_encoder_layers=best_params['num_encoder_layers'],
        num_main_layers=best_params['num_main_layers'],
        dropout=best_params['dropout']
    ).to(device)

    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = 50 * steps_per_epoch  # 50 epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=best_params['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Final training loop with optimizations
    for epoch in range(50):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Move batch to device
            glucose = batch['glucose'].to(device, non_blocking=True)
            carbs = batch['carbs'].to(device, non_blocking=True)
            bolus = batch['bolus'].to(device, non_blocking=True)
            basal = batch['basal'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(glucose, carbs, bolus, basal)
                loss = loss_fn(outputs.squeeze(), targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                glucose = batch['glucose'].to(device, non_blocking=True)
                carbs = batch['carbs'].to(device, non_blocking=True)
                bolus = batch['bolus'].to(device, non_blocking=True)
                basal = batch['basal'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hyperparameters': best_params
            }, checkpoint_path)
        
        # Print progress
        print(f"Epoch {epoch+1}/50")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()

    print("Training completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_parameters': best_params
    }
    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)