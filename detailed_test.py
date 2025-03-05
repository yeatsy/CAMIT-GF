import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import os
import argparse
from model import CAMIT_GF
from preprocess_data import preprocess_data, FORECAST_HORIZON
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Detailed test for CAMIT-GF Model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--data_file', type=str, default='full_patient_dataset.csv',
                      help='Path to the test data')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for testing')
    parser.add_argument('--use_mps', action='store_true',
                      help='Use MPS (Metal Performance Shaders) on Apple Silicon')
    parser.add_argument('--output_dir', type=str, default='test_results',
                      help='Directory to save test results')
    return parser.parse_args()

def set_device(use_mps):
    """Set up the computation device."""
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(checkpoint_path, device):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model parameters from checkpoint
        if 'hyperparameters' in checkpoint:
            params = checkpoint['hyperparameters']
        else:
            params = {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dropout': 0.0
            }
        
        # Create model
        model = CAMIT_GF(
            d_model=params.get('d_model', 64),
            nhead=params.get('nhead', 4),
            num_encoder_layers=params.get('num_encoder_layers', 2),
            dropout=0.0
        ).to(device)
        
        # Load state dict
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def clarke_error_grid(ref_values, pred_values):
    """Calculate Clarke Error Grid zones."""
    # Initialize zone matrix
    zones = np.zeros_like(ref_values, dtype=int)
    
    for i, (ref, pred) in enumerate(zip(ref_values, pred_values)):
        # Zone A: Clinically Accurate
        if (pred <= 70 and ref <= 70) or (pred <= 1.2*ref and pred >= 0.8*ref):
            zones[i] = 0
        # Zone B: Benign Errors
        elif ((ref >= 180 and pred <= ref and pred >= 70) or 
              (ref <= 70 and pred <= 180 and pred >= ref) or
              (ref >= 70 and ref <= 180 and pred >= 0.7*ref and pred <= 1.3*ref)):
            zones[i] = 1
        # Zone C: Overcorrection
        elif ((ref > 180 and pred <= 70) or (ref < 70 and pred >= 180)):
            zones[i] = 2
        # Zone D: Failure to Detect
        elif ((ref >= 70 and pred >= 180 and ref <= 180) or
              (ref <= 180 and pred <= 70 and ref >= 70)):
            zones[i] = 3
        # Zone E: Erroneous Treatment
        else:
            zones[i] = 4
            
    return zones

def plot_clarke_grid(ref_values, pred_values, time_step, ax):
    """Plot Clarke Error Grid for a specific time step."""
    # Set up the plot
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.plot([0, 400], [0, 400], 'k--')
    ax.plot([0, 175/3], [70, 70], 'k-')
    ax.plot([175/3, 400], [70, 400], 'k-')
    ax.plot([70, 70], [84, 400], 'k-')
    ax.plot([0, 70], [180, 180], 'k-')
    ax.plot([70, 400], [180, 180], 'k-')
    ax.plot([0, 70], [0, 70], 'k-')
    
    # Calculate zones
    zones = clarke_error_grid(ref_values, pred_values)
    
    # Plot points with different colors for each zone
    colors = ['green', 'blue', 'yellow', 'orange', 'red']
    zone_names = ['A', 'B', 'C', 'D', 'E']
    
    # Calculate percentages for each zone
    zone_percentages = []
    for zone in range(5):
        pct = np.sum(zones == zone) / len(zones) * 100
        zone_percentages.append(pct)
        
    # Plot points and create legend
    for zone in range(5):
        mask = zones == zone
        if np.any(mask):
            ax.scatter(ref_values[mask], pred_values[mask], 
                      c=colors[zone], alpha=0.5, s=10,
                      label=f'Zone {zone_names[zone]}: {zone_percentages[zone]:.1f}%')
    
    ax.set_title(f't+{time_step*5}min Prediction')
    ax.set_xlabel('Reference Glucose (mg/dL)')
    ax.set_ylabel('Predicted Glucose (mg/dL)')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)

def detailed_test(model, test_loader, device, output_dir):
    """Run detailed test with Clarke Error Grid analysis for each time step."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_original_predictions = []
    all_original_targets = []
    
    print("\nCollecting predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            # Move data to device
            glucose = batch['glucose'].to(device)
            carbs = batch['carbs'].to(device)
            bolus = batch['bolus'].to(device)
            basal = batch['basal'].to(device)
            targets = batch['target']
            
            # Get predictions
            outputs = model(glucose, carbs, bolus, basal)
            
            # Get original scale values
            glucose_means = batch['glucose_mean'].numpy()
            glucose_stds = batch['glucose_std'].numpy()
            
            # Convert predictions and targets back to original scale
            original_predictions = outputs.cpu().numpy() * glucose_stds[:, None] + glucose_means[:, None]
            original_targets = targets.numpy() * glucose_stds[:, None] + glucose_means[:, None]
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_original_predictions.append(original_predictions)
            all_original_targets.append(original_targets)
    
    # Combine all batches
    print("\nProcessing results...")
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    original_predictions = np.concatenate(all_original_predictions, axis=0)
    original_targets = np.concatenate(all_original_targets, axis=0)
    
    # Calculate metrics for each time step
    print("\nCalculating metrics...")
    metrics = []
    for t in tqdm(range(FORECAST_HORIZON), desc="Time steps"):
        rmse = np.sqrt(mean_squared_error(targets[:, t], predictions[:, t]))
        mae = mean_absolute_error(targets[:, t], predictions[:, t])
        r2 = r2_score(targets[:, t], predictions[:, t])
        
        # Calculate original scale RMSE
        orig_rmse = np.sqrt(mean_squared_error(original_targets[:, t], original_predictions[:, t]))
        orig_mae = mean_absolute_error(original_targets[:, t], original_predictions[:, t])
        
        metrics.append({
            'time_step': t,
            'minutes_ahead': (t+1)*5,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Original_RMSE': orig_rmse,
            'Original_MAE': orig_mae
        })
    
    # Create directory for Clarke Error Grid plots
    clarke_grid_dir = os.path.join(output_dir, 'clarke_error_grids')
    os.makedirs(clarke_grid_dir, exist_ok=True)
    
    # Plot individual Clarke Error Grids
    print("\nGenerating Clarke Error Grid plots...")
    for t in tqdm(range(FORECAST_HORIZON), desc="Creating Clarke Error Grids"):
        # Create new figure for each time step
        plt.figure(figsize=(10, 10))
        plot_clarke_grid(original_targets[:, t], original_predictions[:, t], t+1, plt.gca())
        
        # Save individual plot
        filename = f'clarke_grid_t{(t+1)*5:03d}min.png'
        plt.savefig(os.path.join(clarke_grid_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Clarke Error Grid for t+{(t+1)*5}min")
    
    # Plot metrics over time
    print("\nGenerating metrics plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    times = [m['minutes_ahead'] for m in metrics]
    rmse_values = [m['RMSE'] for m in metrics]
    mae_values = [m['MAE'] for m in metrics]
    r2_values = [m['R2'] for m in metrics]
    orig_rmse_values = [m['Original_RMSE'] for m in metrics]
    orig_mae_values = [m['Original_MAE'] for m in metrics]
    
    # Plot normalized metrics
    ax1.plot(times, rmse_values, 'b-o')
    ax1.set_title('Normalized RMSE over Time')
    ax1.set_xlabel('Minutes Ahead')
    ax1.set_ylabel('RMSE (normalized)')
    ax1.grid(True)
    
    ax2.plot(times, mae_values, 'g-o')
    ax2.set_title('Normalized MAE over Time')
    ax2.set_xlabel('Minutes Ahead')
    ax2.set_ylabel('MAE (normalized)')
    ax2.grid(True)
    
    # Plot original scale metrics
    ax3.plot(times, orig_rmse_values, 'r-o')
    ax3.set_title('RMSE over Time (mg/dL)')
    ax3.set_xlabel('Minutes Ahead')
    ax3.set_ylabel('RMSE (mg/dL)')
    ax3.grid(True)
    
    ax4.plot(times, orig_mae_values, 'm-o')
    ax4.set_title('MAE over Time (mg/dL)')
    ax4.set_xlabel('Minutes Ahead')
    ax4.set_ylabel('MAE (mg/dL)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_over_time.png'))
    print(f"Metrics plots saved to {os.path.join(output_dir, 'metrics_over_time.png')}")
    plt.close()
    
    # Save metrics to file
    print("\nSaving detailed metrics...")
    import json
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed metrics saved to {os.path.join(output_dir, 'detailed_metrics.json')}")
    
    # Print summary of original scale metrics
    print("\nTest Results Summary (in mg/dL):")
    print("\nMetrics by prediction horizon:")
    print(f"{'Minutes Ahead':>12} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 44)
    for m in metrics:
        print(f"{m['minutes_ahead']:12d} {m['Original_RMSE']:10.1f} {m['Original_MAE']:10.1f} {m['R2']:10.4f}")
    
    return metrics

def main():
    args = parse_args()
    device = set_device(args.use_mps)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    _, _, test_dataset = preprocess_data(args.data_file)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == 'mps' else 4,
        pin_memory=device.type == 'cuda'
    )
    
    # Load model
    model = load_model(args.checkpoint_path, device)
    
    # Run detailed test
    metrics = detailed_test(model, test_loader, device, args.output_dir)
    
    # Print summary
    print("\nTest Results Summary:")
    print("\nMetrics by prediction horizon:")
    print(f"{'Minutes Ahead':>12} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 44)
    for m in metrics:
        print(f"{m['minutes_ahead']:12d} {m['RMSE']:10.4f} {m['MAE']:10.4f} {m['R2']:10.4f}")
    
    print(f"\nResults saved in {args.output_dir}/")

if __name__ == "__main__":
    main() 