import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
import json
from preprocess_data import FORECAST_HORIZON

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
        # Attention layers with batch_first=True for MPS/CUDA compatibility
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # Simplified: Use a single cross-attention layer instead of three separate ones
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # Simplified gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Sigmoid()
        )
        
        # Feedforward network with reduced capacity
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),  # Reduced from 4*d_model
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Layer scale parameters
        self.ls1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.ls3 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, glucose_hidden, carbs_hidden, bolus_hidden, basal_hidden):
        # Save input for residual connection
        residual = glucose_hidden
        
        # Self-attention on glucose sequence with pre-norm
        glucose_norm = self.norm1(glucose_hidden)
        attn_output, _ = self.self_attn(glucose_norm, glucose_norm, glucose_norm)
        glucose_hidden = residual + self.dropout(self.ls1.unsqueeze(0).unsqueeze(0) * attn_output)
        
        # Save for next residual connection
        residual = glucose_hidden
        
        # Combine all inputs for cross-attention
        # Concatenate along batch dimension for efficient processing
        combined_inputs = torch.cat([carbs_hidden, bolus_hidden, basal_hidden], dim=0)
        
        # Cross-attention with combined inputs
        glucose_norm = self.norm2(glucose_hidden)
        
        # Repeat glucose norm for each input type
        repeated_glucose = glucose_norm.repeat(3, 1, 1)
        
        # Perform cross-attention in a single operation
        combined_attn, _ = self.cross_attn(repeated_glucose, combined_inputs, combined_inputs)
        
        # Split the results back
        carbs_attn, bolus_attn, basal_attn = torch.chunk(combined_attn, 3, dim=0)
        
        # Calculate gates for each input type
        gates = self.gate(glucose_norm)
        carbs_gate = gates[:, :, 0:1]
        bolus_gate = gates[:, :, 1:2]
        basal_gate = gates[:, :, 2:3]
        
        # Apply gates
        fused_attn = (
            carbs_gate * carbs_attn + 
            bolus_gate * bolus_attn + 
            basal_gate * basal_attn
        ) / 3.0  # Normalize
        
        # Apply residual connection
        glucose_hidden = residual + self.dropout(self.ls2.unsqueeze(0).unsqueeze(0) * fused_attn)
        
        # Save for final residual connection
        residual = glucose_hidden
        
        # Feedforward network with pre-norm
        glucose_norm = self.norm3(glucose_hidden)
        ff_output = self.feed_forward(glucose_norm)
        glucose_hidden = residual + self.dropout(self.ls3.unsqueeze(0).unsqueeze(0) * ff_output)

        return glucose_hidden

# Main CAMIT-GF model class
class CAMIT_GF(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=2, dropout=0.2, forecast_horizon=12):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
        # Combined input projection for all features
        self.input_proj = nn.Linear(4, d_model)  # Combined projection for [glucose, carbs, bolus, basal]
        
        # Simplified time embedding
        self.time_embedding = nn.Embedding(288, d_model)  # 288 = 24 hours * 12 (5-min intervals)
        
        # Single positional encoding layer
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Single layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Single unified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Temporal modeling with GRU (simpler than LSTM)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,  # Using 2 layers for better temporal modeling
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        # Simplified prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),  # Add normalization before final projection
            nn.Linear(d_model, forecast_horizon)
        )
        
        # Simple residual connection
        self.residual_proj = nn.Linear(1, forecast_horizon)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization with smaller gain"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)  # Reduced gain for more stable training
            elif p.dim() == 1:
                nn.init.zeros_(p)  # Initialize biases to zero
    
    def forward(self, glucose, carbs, bolus, basal, time_indices=None):
        # Combine inputs
        x = torch.cat([glucose, carbs, bolus, basal], dim=-1)  # [batch, seq_len, 4]
        
        # Project combined inputs
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add time embedding if provided
        if time_indices is not None:
            time_emb = self.time_embedding(time_indices)
            x = x + time_emb
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Apply transformer encoder
        x = self.encoder(x)
        
        # Apply GRU for temporal modeling
        x, _ = self.gru(x)
        
        # Take the last sequence element for prediction
        x = x[:, -1, :]  # [batch, d_model]
        
        # Generate prediction
        pred = self.prediction_head(x)  # [batch, forecast_horizon]
        
        # Add residual connection from last glucose value
        res = self.residual_proj(glucose[:, -1, :])  # Project last glucose value
        pred = pred + res
        
        return pred

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CAMIT-GF Training')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training')
    parser.add_argument('--use_mps', action='store_true', help='Use MPS (Metal Performance Shaders) on Apple Silicon')
    args = parser.parse_args()

    # Set device based on available hardware
    if args.use_mps:
        # More robust MPS detection
        try:
            import torch.backends.mps
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
        # Set up CUDA and optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load and preprocess actual data
    from preprocess_data import preprocess_data
    train_dataset, val_dataset, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create data loaders with optimizations
    num_workers = 0 if device.type == 'mps' else (4 if device.type == 'cuda' else 2)
    pin_memory = device.type == 'cuda'  # Only pin memory for CUDA
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Enable automatic mixed precision if requested and on CUDA
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Hyperparameter search space
    param_grid = {
        'd_model': [32, 64, 128],
        'nhead': [4, 8],
        'num_encoder_layers': [1, 2, 3],
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
            dropout=params['dropout']
        ).to(device)
        
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
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
                
                # Forward pass with or without mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(glucose, carbs, bolus, basal)
                        loss = loss_fn(outputs.squeeze(), targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = loss_fn(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                
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
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(glucose, carbs, bolus, basal)
                            loss = loss_fn(outputs.squeeze(), targets)
                    else:
                        outputs = model(glucose, carbs, bolus, basal)
                        loss = loss_fn(outputs.squeeze(), targets)
                    
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            
            print(f"Epoch {epoch+1}/{num_search_epochs} - Val Loss: {val_loss:.4f}")
            
            # Clear cache after each epoch
            if device.type == 'cuda':
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
        dropout=best_params['dropout']
    ).to(device)

    # Enable multi-GPU training if available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
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
            
            # Forward pass with or without mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(glucose, carbs, bolus, basal)
                    loss = loss_fn(outputs.squeeze(), targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(glucose, carbs, bolus, basal)
                loss = loss_fn(outputs.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
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
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(glucose, carbs, bolus, basal)
                        loss = loss_fn(outputs.squeeze(), targets)
                else:
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
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
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
        if device.type == 'cuda':
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