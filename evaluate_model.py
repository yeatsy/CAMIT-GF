import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from model import CAMIT_GF
from preprocess_data import preprocess_data, FORECAST_HORIZON
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CAMIT-GF model on predicting 12 steps into the future')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample predictions to visualize')
    parser.add_argument('--use_mps', action='store_true', help='Use MPS (Metal Performance Shaders) on Apple Silicon')
    return parser.parse_args()

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

def evaluate_model(model, test_loader, device, num_samples=10, output_dir='evaluation_results'):
    """Evaluate model performance on test data"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics
    all_predictions = []
    all_targets = []
    all_original_predictions = []
    all_original_targets = []
    
    # Sample predictions for visualization
    sample_indices = np.random.choice(len(test_loader.dataset), min(num_samples, len(test_loader.dataset)), replace=False)
    samples_collected = 0
    sample_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            glucose = batch['glucose'].to(device, non_blocking=True)
            carbs = batch['carbs'].to(device, non_blocking=True)
            bolus = batch['bolus'].to(device, non_blocking=True)
            basal = batch['basal'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            original_glucose = batch['original_glucose']
            original_target = batch['original_target']
            glucose_mean = batch['glucose_mean']
            glucose_std = batch['glucose_std']
            
            # Generate time-of-day indices
            time_indices = generate_time_indices(glucose.size(0), glucose.size(1), device)
            
            # Forward pass
            outputs = model(glucose, carbs, bolus, basal, time_indices)
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.unsqueeze(1).expand(-1, FORECAST_HORIZON).cpu().numpy())
            
            # Convert standardized predictions back to original scale
            original_predictions = []
            for i in range(outputs.size(0)):
                # Unstandardize: pred * std + mean
                orig_pred = outputs[i].cpu().numpy() * glucose_std[i].item() + glucose_mean[i].item()
                original_predictions.append(orig_pred)
            
            all_original_predictions.extend(original_predictions)
            all_original_targets.extend([original_target[i].item()] * FORECAST_HORIZON for i in range(len(original_target)))
            
            # Collect sample data for visualization
            if samples_collected < num_samples:
                for i in range(outputs.size(0)):
                    global_idx = batch_idx * test_loader.batch_size + i
                    if global_idx in sample_indices:
                        sample_data.append({
                            'glucose_history': original_glucose[i].squeeze().cpu().numpy(),
                            'prediction': original_predictions[i],
                            'target': original_target[i].item(),
                            'time_indices': time_indices[i].cpu().numpy()
                        })
                        samples_collected += 1
                        if samples_collected >= num_samples:
                            break
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics for each time step
    mse_per_step = []
    mae_per_step = []
    rmse_per_step = []
    
    for step in range(FORECAST_HORIZON):
        step_preds = np.array([pred[step] for pred in all_original_predictions])
        step_targets = np.array([target[step] for target in all_original_targets])
        
        mse = mean_squared_error(step_targets, step_preds)
        mae = mean_absolute_error(step_targets, step_preds)
        rmse = np.sqrt(mse)
        
        mse_per_step.append(mse)
        mae_per_step.append(mae)
        rmse_per_step.append(rmse)
    
    # Overall metrics
    overall_mse = mean_squared_error(
        np.array(all_original_targets).flatten(), 
        np.array(all_original_predictions).flatten()
    )
    overall_mae = mean_absolute_error(
        np.array(all_original_targets).flatten(), 
        np.array(all_original_predictions).flatten()
    )
    overall_rmse = np.sqrt(overall_mse)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Step': list(range(1, FORECAST_HORIZON + 1)),
        'MSE': mse_per_step,
        'MAE': mae_per_step,
        'RMSE': rmse_per_step
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_per_step.csv'), index=False)
    
    # Save overall metrics
    with open(os.path.join(output_dir, 'overall_metrics.txt'), 'w') as f:
        f.write(f"Overall MSE: {overall_mse:.4f}\n")
        f.write(f"Overall MAE: {overall_mae:.4f}\n")
        f.write(f"Overall RMSE: {overall_rmse:.4f}\n")
    
    # Plot metrics by time step
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(range(1, FORECAST_HORIZON + 1), mse_per_step, marker='o')
    plt.title('MSE by Prediction Step')
    plt.xlabel('Steps into Future (5-min intervals)')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, FORECAST_HORIZON + 1), mae_per_step, marker='o', color='orange')
    plt.title('MAE by Prediction Step')
    plt.xlabel('Steps into Future (5-min intervals)')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, FORECAST_HORIZON + 1), rmse_per_step, marker='o', color='green')
    plt.title('RMSE by Prediction Step')
    plt.xlabel('Steps into Future (5-min intervals)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_step.png'))
    
    # Visualize sample predictions
    for i, sample in enumerate(sample_data):
        plt.figure(figsize=(12, 6))
        
        # Plot history
        history_x = np.arange(-len(sample['glucose_history']), 0)
        plt.plot(history_x, sample['glucose_history'], label='Historical Glucose', color='blue')
        
        # Plot prediction
        future_x = np.arange(1, FORECAST_HORIZON + 1)
        plt.plot(future_x, sample['prediction'], label='Predicted Glucose', color='red', marker='o')
        
        # Plot target (as a horizontal line extending from the last historical point)
        plt.axhline(y=sample['target'], color='green', linestyle='--', 
                   xmin=(len(history_x)) / (len(history_x) + len(future_x)),
                   label='Actual Future Glucose')
        
        # Add vertical line at prediction start
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title(f'Sample Prediction {i+1}')
        plt.xlabel('Time Steps (5-min intervals)')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.legend()
        plt.grid(True)
        
        # Annotate time of day
        time_labels = []
        for t in sample['time_indices']:
            hours = (t // 12) % 24
            minutes = (t % 12) * 5
            time_labels.append(f"{hours:02d}:{minutes:02d}")
        
        # Add time labels for a few points
        for idx in range(0, len(history_x), len(history_x) // 4):
            plt.annotate(time_labels[idx], (history_x[idx], sample['glucose_history'][idx]),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_prediction_{i+1}.png'))
        plt.close()
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    return overall_rmse

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get hyperparameters from checkpoint
    hyperparams = checkpoint['hyperparams']
    
    # Create model
    model = CAMIT_GF(
        d_model=hyperparams['d_model'],
        nhead=hyperparams['nhead'],
        num_encoder_layers=hyperparams['num_encoder_layers'],
        num_main_layers=hyperparams['num_main_layers'],
        dropout=hyperparams['dropout'],
        forecast_horizon=FORECAST_HORIZON
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    print("Loading and preprocessing data...")
    _, _, test_dataset = preprocess_data('full_patient_dataset.csv')
    
    # Create data loader
    num_workers = 0 if device.type == 'mps' else args.num_workers  # MPS doesn't work well with multiple workers
    pin_memory = device.type == 'cuda'  # Only pin memory for CUDA
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Evaluate model
    print(f"Evaluating model on {len(test_dataset)} test samples...")
    rmse = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print(f"Overall RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main() 