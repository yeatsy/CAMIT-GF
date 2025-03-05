import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import CAMIT_GF
from preprocess_data import preprocess_data, SEQ_LEN, FORECAST_HORIZON
from torch.utils.data import DataLoader
import seaborn as sns
import multiprocessing
import time

def parse_args():
    parser = argparse.ArgumentParser(description='CAMIT-GF Model Testing')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pt', 
                        help='Path to the model checkpoint')
    parser.add_argument('--data_file', type=str, default='full_patient_dataset.csv', 
                        help='Path to the patient data CSV file')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='test_results', 
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of sample predictions to visualize')
    parser.add_argument('--use_mps', action='store_true', 
                        help='Use MPS (Metal Performance Shaders) on Apple Silicon')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save all predictions to a CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--max_eval_batches', type=int, default=100,
                        help='Maximum number of batches to evaluate (for faster testing)')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Run in fast mode with limited evaluation')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for data loading (default: auto)')
    return parser.parse_args()

def set_device(use_mps=False):
    """Set the appropriate device for inference."""
    if use_mps and torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model hyperparameters from checkpoint
        if 'hyperparameters' in checkpoint:
            args = checkpoint['hyperparameters']
        else:
            args = checkpoint.get('args', {})
            
        d_model = args.get('d_model', 64)
        nhead = args.get('nhead', 4)
        num_encoder_layers = args.get('num_encoder_layers', 2)
        dropout = args.get('dropout', 0.1)  # Add dropout parameter with default
        
        print(f"Model hyperparameters:")
        print(f"- d_model: {d_model}")
        print(f"- nhead: {nhead}")
        print(f"- num_encoder_layers: {num_encoder_layers}")
        print(f"- dropout: {dropout}")
        
        # Create model with the same architecture
        model = CAMIT_GF(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=0.0  # No dropout for inference
        ).to(device)
        
        # Check if the checkpoint contains model_state_dict or just the state dict directly
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            # Try to load the state dict directly
            model_state = checkpoint
            
        # Load model weights with error handling for incompatible keys
        incompatible_keys = model.load_state_dict(model_state, strict=False)
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in state dict: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {incompatible_keys.unexpected_keys}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def evaluate_model(model, test_loader, device, max_eval_batches=100):
    """Evaluate the model on test data."""
    model.eval()
    all_targets = []
    all_predictions = []
    
    # Store full sequences for visualization
    sequence_samples = []
    num_sequences_to_store = 12  # Store 12 sequences for visualization
    
    # Track errors during evaluation
    evaluation_errors = 0
    
    # Flag to indicate if we've collected enough sequences
    sequences_collected = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Break after max_eval_batches for faster evaluation
            if batch_idx >= max_eval_batches and sequences_collected:
                print(f"Reached maximum evaluation batches ({max_eval_batches}). Stopping evaluation.")
                break
                
            try:
                # Move data to device
                glucose = batch['glucose'].to(device, non_blocking=True)
                carbs = batch['carbs'].to(device, non_blocking=True)
                bolus = batch['bolus'].to(device, non_blocking=True)
                basal = batch['basal'].to(device, non_blocking=True)
                targets = batch['target'].cpu().numpy()
                
                # Get original glucose values for visualization
                original_glucose = batch['original_glucose'].cpu().numpy()
                original_targets = batch['original_target'].cpu().numpy()
                glucose_means = batch['glucose_mean'].cpu().numpy()
                glucose_stds = batch['glucose_std'].cpu().numpy()
                
                # Generate predictions for all forecast horizons
                outputs = model(glucose, carbs, bolus, basal)
                outputs = outputs.cpu().numpy()
                
                # Store results for metrics
                all_targets.extend(targets)
                all_predictions.extend(outputs[:, -1])  # Use last prediction for metrics
                
                # Only generate sequences for visualization if we haven't collected enough yet
                if not sequences_collected and len(sequence_samples) < num_sequences_to_store:
                    # Only process a small batch for sequence generation (for speed)
                    small_batch_size = min(4, glucose.size(0))
                    
                    # Process a few samples for visualization
                    for i in range(small_batch_size):
                        if len(sequence_samples) >= num_sequences_to_store:
                            sequences_collected = True
                            break
                            
                        # Get individual sample
                        sample_glucose = glucose[i:i+1]
                        sample_carbs = carbs[i:i+1]
                        sample_bolus = bolus[i:i+1]
                        sample_basal = basal[i:i+1]
                        sample_target = targets[i]
                        
                        # Get original values for this sample
                        sample_original_glucose = original_glucose[i]
                        sample_original_target = original_targets[i]
                        sample_glucose_mean = glucose_means[i]
                        sample_glucose_std = glucose_stds[i]
                        
                        # Get predictions for all horizons
                        hour_predictions = outputs[i]
                        
                        # Convert normalized predictions back to original scale
                        original_predictions = hour_predictions * sample_glucose_std + sample_glucose_mean
                        
                        # Store the sequence, prediction sequence, and target
                        sequence_samples.append({
                            'glucose_seq': sample_glucose.cpu().numpy().flatten(),
                            'prediction_seq': hour_predictions,
                            'target': sample_target,
                            'original_glucose_seq': sample_original_glucose.flatten(),
                            'original_prediction_seq': original_predictions,
                            'original_target': sample_original_target,
                            'glucose_mean': sample_glucose_mean,
                            'glucose_std': sample_glucose_std
                        })
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                evaluation_errors += 1
                continue
    
    if evaluation_errors > 0:
        print(f"Warning: {evaluation_errors} batches had errors during evaluation")
    
    if len(all_targets) == 0:
        print("Error: No valid predictions were made. Check your data and model.")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'mape': float('nan'),
            'targets': np.array([]),
            'predictions': np.array([])
        }
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
    
    # Calculate Clarke Error Grid Analysis metrics if available
    try:
        from sklearn.metrics import confusion_matrix
        
        # Define zones for Clarke Error Grid (simplified)
        def clarke_error_grid_zones(true_values, pred_values):
            zones = np.zeros_like(true_values, dtype=int)
            for i, (true, pred) in enumerate(zip(true_values, pred_values)):
                if (pred <= 70 and true <= 70) or (pred <= 1.2*true and pred >= 0.8*true):
                    zones[i] = 0  # Zone A
                elif (pred >= 180 and true <= 70) or (pred <= 70 and true >= 180):
                    zones[i] = 4  # Zone E
                elif ((pred >= 70 and pred <= 180) and true <= 70) or ((true >= 70 and true <= 180) and pred <= 70):
                    zones[i] = 3  # Zone D
                elif (pred <= 1.2*true and pred >= true) or (pred <= true and pred >= 0.8*true):
                    zones[i] = 1  # Zone B
                else:
                    zones[i] = 2  # Zone C
            return zones
        
        zones = clarke_error_grid_zones(all_targets, all_predictions)
        zone_counts = np.bincount(zones, minlength=5)
        zone_percentages = zone_counts / len(zones) * 100
        
        clarke_grid_results = {
            'zone_A_percent': zone_percentages[0],
            'zone_B_percent': zone_percentages[1],
            'zone_C_percent': zone_percentages[2],
            'zone_D_percent': zone_percentages[3],
            'zone_E_percent': zone_percentages[4],
        }
    except Exception as e:
        print(f"Warning: Could not calculate Clarke Error Grid metrics: {e}")
        clarke_grid_results = {}
    
    print(f"Evaluation completed on {len(all_targets)} samples")
    print(f"Generated {len(sequence_samples)} sequence visualizations")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'targets': all_targets,
        'predictions': all_predictions,
        'sequence_samples': sequence_samples,  # Add the sequence samples
        **clarke_grid_results
    }

def visualize_predictions(results, num_samples, output_dir):
    """Visualize sample predictions vs targets."""
    targets = results['targets']
    predictions = results['predictions']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select samples to visualize
    indices = np.random.choice(len(targets), min(num_samples, len(targets)), replace=False)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot([targets[idx]], [0], 'bo', label='Actual', markersize=8)
        ax.plot([predictions[idx]], [0], 'ro', label='Predicted', markersize=8)
        ax.set_title(f'Sample {i+1}: Actual={targets[idx]:.2f}, Predicted={predictions[idx]:.2f}, Error={predictions[idx]-targets[idx]:.2f}')
        ax.set_yticks([])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    print(f"Saved sample predictions visualization to {os.path.join(output_dir, 'sample_predictions.png')}")
    
    # Create scatter plot of predictions vs targets
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(targets), p(np.sort(targets)), "b--", alpha=0.7, 
             label=f"Regression Line (y = {z[0]:.4f}x + {z[1]:.4f})")
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
    print(f"Saved scatter plot to {os.path.join(output_dir, 'scatter_plot.png')}")
    
    # Create error histogram
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
    print(f"Saved error histogram to {os.path.join(output_dir, 'error_histogram.png')}")
    
    # Create Clarke Error Grid if possible
    try:
        plt.figure(figsize=(10, 10))
        
        # Define the boundaries of the Clarke Error Grid
        plt.plot([0, 400], [0, 400], 'k--')
        plt.plot([0, 175/3], [70, 70], 'k-')
        plt.plot([175/3, 400], [70, 400], 'k-')
        plt.plot([70, 70], [0, 175/3], 'k-')
        plt.plot([70, 400], [175/3, 400], 'k-')
        plt.plot([0, 70], [180, 180], 'k-')
        plt.plot([70, 290], [180, 400], 'k-')
        plt.plot([180, 180], [0, 70], 'k-')
        plt.plot([180, 400], [70, 290], 'k-')
        
        # Plot the data points
        plt.scatter(targets, predictions, alpha=0.5)
        
        # Add labels and title
        plt.xlabel('Reference Glucose (mg/dL)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        plt.title('Clarke Error Grid Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add zone labels
        plt.text(30, 15, 'A', fontsize=15)
        plt.text(370, 260, 'B', fontsize=15)
        plt.text(280, 380, 'B', fontsize=15)
        plt.text(160, 370, 'C', fontsize=15)
        plt.text(160, 15, 'C', fontsize=15)
        plt.text(30, 140, 'D', fontsize=15)
        plt.text(370, 120, 'D', fontsize=15)
        plt.text(30, 370, 'E', fontsize=15)
        plt.text(370, 30, 'E', fontsize=15)
        
        plt.xlim(0, 400)
        plt.ylim(0, 400)
        
        plt.savefig(os.path.join(output_dir, 'clarke_error_grid.png'))
        print(f"Saved Clarke Error Grid to {os.path.join(output_dir, 'clarke_error_grid.png')}")
    except Exception as e:
        print(f"Warning: Could not create Clarke Error Grid visualization: {e}")
        
    # Create a summary plot with key metrics
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.8, f"RMSE: {results['rmse']:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.7, f"MAE: {results['mae']:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.6, f"R²: {results['r2']:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.5, f"MAPE: {results['mape']:.2f}%", fontsize=14, ha='center')
        
        if 'zone_A_percent' in results:
            plt.text(0.5, 0.4, "Clarke Error Grid Analysis:", fontsize=14, ha='center')
            plt.text(0.5, 0.3, f"Zone A: {results['zone_A_percent']:.2f}%", fontsize=12, ha='center')
            plt.text(0.5, 0.25, f"Zone B: {results['zone_B_percent']:.2f}%", fontsize=12, ha='center')
            plt.text(0.5, 0.2, f"Zone C: {results['zone_C_percent']:.2f}%", fontsize=12, ha='center')
            plt.text(0.5, 0.15, f"Zone D: {results['zone_D_percent']:.2f}%", fontsize=12, ha='center')
            plt.text(0.5, 0.1, f"Zone E: {results['zone_E_percent']:.2f}%", fontsize=12, ha='center')
        
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
        print(f"Saved metrics summary to {os.path.join(output_dir, 'metrics_summary.png')}")
    except Exception as e:
        print(f"Warning: Could not create metrics summary visualization: {e}")

def visualize_sequences(results, output_dir):
    """Visualize the 12 predicted sequences for the hour-ahead forecast."""
    sequence_samples = results.get('sequence_samples', [])
    
    if not sequence_samples:
        print("No sequence samples available for visualization.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure for all sequences
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    # Set a consistent style
    sns.set_style("whitegrid")
    
    # Shuffle the sequences to get different ones
    np.random.shuffle(sequence_samples)
    
    for i, sample in enumerate(sequence_samples):
        if i >= 12:  # Limit to 12 sequences
            break
            
        ax = axes[i]
        
        # Get the data - use original glucose values
        original_glucose_seq = sample['original_glucose_seq']
        original_prediction_seq = sample['original_prediction_seq']
        original_target = sample['original_target']
        
        # Time points for x-axis (assuming 5-minute intervals)
        historical_time = np.arange(0, len(original_glucose_seq) * 5, 5)  # in minutes
        forecast_time = np.arange(historical_time[-1] + 5, historical_time[-1] + 5 + len(original_prediction_seq) * 5, 5)
        
        # Plot the historical glucose sequence
        ax.plot(historical_time, original_glucose_seq, 'b-', label='Historical Glucose')
        
        # Plot the predicted sequence for the hour ahead
        ax.plot(forecast_time, original_prediction_seq, 'r-', label='Predicted Sequence')
        
        # Plot the actual target point at the end of the hour
        ax.plot([forecast_time[-1]], [original_target], 'go', markersize=8, label='Actual Target')
        
        # Add labels and title
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Glucose Level (mg/dL)')
        ax.set_title(f'Sequence {i+1}: Final Pred={original_prediction_seq[-1]:.1f}, Actual={original_target:.1f}')
        
        # Add a vertical line at the forecast point
        ax.axvline(x=historical_time[-1], color='k', linestyle='--', alpha=0.5)
        
        # Add text annotation for the error
        error = original_prediction_seq[-1] - original_target
        ax.text(0.05, 0.05, f'Final Error: {error:.1f} mg/dL', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Only add legend to the first plot to save space
        if i == 0:
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_predictions.png'))
    print(f"Saved sequence predictions visualization to {os.path.join(output_dir, 'sequence_predictions.png')}")
    
    # Create a more detailed view of each sequence individually
    for i, sample in enumerate(sequence_samples):
        if i >= 12:  # Limit to 12 sequences
            break
            
        plt.figure(figsize=(12, 6))
        
        # Get the data - use original glucose values
        original_glucose_seq = sample['original_glucose_seq']
        original_prediction_seq = sample['original_prediction_seq']
        original_target = sample['original_target']
        
        # Time points for x-axis (assuming 5-minute intervals)
        historical_time = np.arange(0, len(original_glucose_seq) * 5, 5)  # in minutes
        forecast_time = np.arange(historical_time[-1] + 5, historical_time[-1] + 5 + len(original_prediction_seq) * 5, 5)
        
        # Plot the historical glucose sequence
        plt.plot(historical_time, original_glucose_seq, 'b-', linewidth=2, label='Historical Glucose')
        
        # Plot the predicted sequence for the hour ahead
        plt.plot(forecast_time, original_prediction_seq, 'r-', linewidth=2, label='Predicted Sequence')
        
        # Plot the actual target point at the end of the hour
        plt.plot([forecast_time[-1]], [original_target], 'go', markersize=10, label='Actual Target')
        
        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.title(f'Sequence {i+1}: Hour-Ahead Prediction Sequence')
        
        # Add a vertical line at the forecast point
        plt.axvline(x=historical_time[-1], color='k', linestyle='--', alpha=0.5, 
                   label='Forecast Start')
        
        # Add text annotation for the error
        error = original_prediction_seq[-1] - original_target
        plt.text(0.05, 0.05, f'Final Error: {error:.1f} mg/dL', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Add annotations for each prediction point
        for j, pred in enumerate(original_prediction_seq):
            plt.annotate(f'{pred:.1f}', 
                        (forecast_time[j], pred),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
        
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save the individual sequence plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sequence_{i+1}_detail.png'))
        plt.close()
    
    print(f"Saved {min(len(sequence_samples), 12)} individual sequence visualizations to {output_dir}")
    
    # Create a combined plot showing all 12 sequences in one timeline
    plt.figure(figsize=(15, 8))
    
    # Time points for x-axis (assuming 5-minute intervals)
    historical_time = np.arange(0, SEQ_LEN * 5, 5)  # in minutes
    forecast_time = np.arange(historical_time[-1] + 5, historical_time[-1] + 5 + FORECAST_HORIZON * 5, 5)
    
    # Plot each sequence
    for i, sample in enumerate(sequence_samples):
        if i >= 12:  # Limit to 12 sequences
            break
            
        # Get the data - use original glucose values
        original_glucose_seq = sample['original_glucose_seq']
        original_prediction_seq = sample['original_prediction_seq']
        original_target = sample['original_target']
        
        # Plot with different colors and alpha for visibility
        plt.plot(historical_time, original_glucose_seq, '-', alpha=0.5)
        plt.plot(forecast_time, original_prediction_seq, '-', alpha=0.7)
        plt.plot([forecast_time[-1]], [original_target], 'o', markersize=6, alpha=0.7)
    
    # Add a vertical line at the forecast point
    plt.axvline(x=historical_time[-1], color='k', linestyle='--', alpha=0.5, label='Forecast Start')
    
    # Add labels and title
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.title('All 12 Sequences with Hour-Ahead Predictions')
    
    # Add legend
    plt.plot([], [], 'b-', label='Historical Glucose')
    plt.plot([], [], 'r-', label='Predicted Sequence')
    plt.plot([], [], 'go', label='Actual Target')
    plt.legend(loc='best')
    
    plt.grid(True, alpha=0.3)
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_sequences_combined.png'))
    print(f"Saved combined sequences visualization to {os.path.join(output_dir, 'all_sequences_combined.png')}")
    
    # Create a heatmap visualization of all prediction sequences
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    prediction_matrix = np.zeros((len(sequence_samples[:12]), FORECAST_HORIZON))
    for i, sample in enumerate(sequence_samples[:12]):
        prediction_matrix[i] = sample['original_prediction_seq']
    
    # Create heatmap
    ax = sns.heatmap(prediction_matrix, cmap="viridis", 
                    xticklabels=[f"{t} min" for t in forecast_time],
                    yticklabels=[f"Seq {i+1}" for i in range(len(prediction_matrix))],
                    annot=True, fmt=".1f")
    
    plt.title('Heatmap of Predicted Glucose Values (mg/dL) for 12 Sequences')
    plt.xlabel('Time (minutes from forecast start)')
    plt.ylabel('Sequence')
    
    # Save the heatmap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_heatmap.png'))
    print(f"Saved prediction heatmap to {os.path.join(output_dir, 'prediction_heatmap.png')}")

def analyze_model_performance(results, output_dir):
    """
    Analyze why the model isn't performing well and generate diagnostic plots.
    """
    print("\n=== Model Performance Analysis ===")
    
    # Extract data
    targets = results['targets']
    predictions = results['predictions']
    sequence_samples = results.get('sequence_samples', [])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Check for bias in predictions
    mean_target = np.mean(targets)
    mean_prediction = np.mean(predictions)
    bias = mean_prediction - mean_target
    
    print(f"Mean target value: {mean_target:.4f}")
    print(f"Mean prediction value: {mean_prediction:.4f}")
    print(f"Prediction bias: {bias:.4f}")
    
    # 2. Check for correlation between predictions and targets
    correlation = np.corrcoef(targets, predictions)[0, 1]
    print(f"Correlation between targets and predictions: {correlation:.4f}")
    
    # 3. Check for range of predictions vs targets
    target_range = (np.min(targets), np.max(targets))
    prediction_range = (np.min(predictions), np.max(predictions))
    
    print(f"Target range: {target_range[0]:.4f} to {target_range[1]:.4f}")
    print(f"Prediction range: {prediction_range[0]:.4f} to {prediction_range[1]:.4f}")
    
    # 4. Check if model is predicting close to the mean
    target_std = np.std(targets)
    prediction_std = np.std(predictions)
    
    print(f"Target standard deviation: {target_std:.4f}")
    print(f"Prediction standard deviation: {prediction_std:.4f}")
    
    if prediction_std < 0.5 * target_std:
        print("WARNING: Model predictions have much lower variance than targets.")
        print("This suggests the model may be defaulting to predicting values close to the mean.")
    
    # 5. Check for patterns in errors
    errors = predictions - targets
    plt.figure(figsize=(12, 8))
    
    # Plot errors vs targets
    plt.subplot(2, 2, 1)
    plt.scatter(targets, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Target Values')
    plt.ylabel('Prediction Errors')
    plt.title('Errors vs Target Values')
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(2, 2, 2)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='-')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (Mean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot predictions vs targets
    plt.subplot(2, 2, 3)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Targets (R²: {results["r2"]:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot error by sequence index (to check for time-dependent patterns)
    plt.subplot(2, 2, 4)
    plt.plot(range(len(errors)), errors, 'b-', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Sample Index')
    plt.ylabel('Error')
    plt.title('Errors by Sample Index')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'))
    print(f"Saved error analysis to {os.path.join(output_dir, 'error_analysis.png')}")
    
    # 6. Analyze sequence predictions
    if sequence_samples:
        # Check how predictions evolve over the sequence
        plt.figure(figsize=(14, 8))
        
        # Get a few sequences
        num_to_show = min(8, len(sequence_samples))
        
        for i in range(num_to_show):
            sample = sequence_samples[i]
            
            # Get original values
            original_glucose_seq = sample['original_glucose_seq']
            original_prediction_seq = sample['original_prediction_seq']
            original_target = sample['original_target']
            
            # Calculate error progression
            last_historical = original_glucose_seq[-1]
            error_progression = np.abs(original_prediction_seq - original_target)
            
            plt.subplot(2, 4, i+1)
            plt.plot(range(len(error_progression)), error_progression, 'r-', marker='o')
            plt.axhline(y=np.abs(last_historical - original_target), color='b', linestyle='--', 
                       label='Baseline Error')
            plt.title(f'Sequence {i+1} Error Progression')
            plt.xlabel('Prediction Step')
            plt.ylabel('Absolute Error (mg/dL)')
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_error_progression.png'))
        print(f"Saved sequence error progression to {os.path.join(output_dir, 'sequence_error_progression.png')}")
        
        # Check if predictions are just following the last observed value
        baseline_errors = []
        model_errors = []
        
        for sample in sequence_samples:
            last_historical = sample['original_glucose_seq'][-1]
            final_prediction = sample['original_prediction_seq'][-1]
            actual_target = sample['original_target']
            
            baseline_error = np.abs(last_historical - actual_target)
            model_error = np.abs(final_prediction - actual_target)
            
            baseline_errors.append(baseline_error)
            model_errors.append(model_error)
        
        baseline_mae = np.mean(baseline_errors)
        model_mae = np.mean(model_errors)
        
        print(f"Baseline MAE (predicting last observed value): {baseline_mae:.4f}")
        print(f"Model MAE: {model_mae:.4f}")
        
        if model_mae > baseline_mae:
            print("WARNING: Model performs worse than simply predicting the last observed value.")
        
        # Visualize baseline vs model errors
        plt.figure(figsize=(10, 6))
        plt.bar(['Baseline (Last Value)', 'Model'], [baseline_mae, model_mae], color=['blue', 'red'])
        plt.ylabel('Mean Absolute Error (mg/dL)')
        plt.title('Model vs Baseline Error Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'baseline_comparison.png'))
        print(f"Saved baseline comparison to {os.path.join(output_dir, 'baseline_comparison.png')}")
    
    print("=== End of Analysis ===\n")
    
    return {
        'bias': bias,
        'correlation': correlation,
        'target_range': target_range,
        'prediction_range': prediction_range,
        'target_std': target_std,
        'prediction_std': prediction_std
    }

def evaluate_baseline_models(test_loader, device, max_eval_batches=100):
    """
    Evaluate simple baseline models for comparison.
    
    Baseline models:
    1. Predict the last observed glucose value
    2. Linear extrapolation from the last two observed values
    3. Mean of the last hour of observed values
    """
    print("\nEvaluating baseline models...")
    
    all_targets = []
    last_value_predictions = []
    linear_extrapolation_predictions = []
    last_hour_mean_predictions = []
    
    # Track errors during evaluation
    evaluation_errors = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating baselines")):
            # Break after max_eval_batches for faster evaluation
            if batch_idx >= max_eval_batches:
                print(f"Reached maximum evaluation batches ({max_eval_batches}). Stopping evaluation.")
                break
                
            try:
                # Get data
                glucose = batch['glucose'].cpu().numpy()
                targets = batch['target'].cpu().numpy()
                
                # Get original glucose values
                original_glucose = batch['original_glucose'].cpu().numpy()
                original_targets = batch['original_target'].cpu().numpy()
                
                batch_size = glucose.shape[0]
                
                for i in range(batch_size):
                    # Get individual sample
                    sample_glucose = glucose[i].flatten()
                    sample_original_glucose = original_glucose[i].flatten()
                    sample_target = targets[i]
                    sample_original_target = original_targets[i]
                    
                    # Baseline 1: Last observed value
                    last_value_pred = sample_glucose[-1]
                    
                    # Baseline 2: Linear extrapolation
                    if len(sample_glucose) >= 2:
                        # Calculate slope from last two points
                        slope = sample_glucose[-1] - sample_glucose[-2]
                        # Extrapolate FORECAST_HORIZON steps ahead
                        linear_extrapolation_pred = sample_glucose[-1] + slope * FORECAST_HORIZON
                    else:
                        linear_extrapolation_pred = sample_glucose[-1]
                    
                    # Baseline 3: Mean of last hour (12 points)
                    last_hour_mean_pred = np.mean(sample_glucose[-min(12, len(sample_glucose)):])
                    
                    # Store results
                    all_targets.append(sample_target)
                    last_value_predictions.append(last_value_pred)
                    linear_extrapolation_predictions.append(linear_extrapolation_pred)
                    last_hour_mean_predictions.append(last_hour_mean_pred)
                    
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
                evaluation_errors += 1
                continue
    
    if evaluation_errors > 0:
        print(f"Warning: {evaluation_errors} batches had errors during baseline evaluation")
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    last_value_predictions = np.array(last_value_predictions)
    linear_extrapolation_predictions = np.array(linear_extrapolation_predictions)
    last_hour_mean_predictions = np.array(last_hour_mean_predictions)
    
    # Calculate metrics for each baseline
    baseline_results = {}
    
    # Last value baseline
    last_value_rmse = np.sqrt(mean_squared_error(all_targets, last_value_predictions))
    last_value_mae = mean_absolute_error(all_targets, last_value_predictions)
    last_value_r2 = r2_score(all_targets, last_value_predictions)
    
    baseline_results['last_value'] = {
        'rmse': last_value_rmse,
        'mae': last_value_mae,
        'r2': last_value_r2,
        'predictions': last_value_predictions
    }
    
    # Linear extrapolation baseline
    linear_extrapolation_rmse = np.sqrt(mean_squared_error(all_targets, linear_extrapolation_predictions))
    linear_extrapolation_mae = mean_absolute_error(all_targets, linear_extrapolation_predictions)
    linear_extrapolation_r2 = r2_score(all_targets, linear_extrapolation_predictions)
    
    baseline_results['linear_extrapolation'] = {
        'rmse': linear_extrapolation_rmse,
        'mae': linear_extrapolation_mae,
        'r2': linear_extrapolation_r2,
        'predictions': linear_extrapolation_predictions
    }
    
    # Last hour mean baseline
    last_hour_mean_rmse = np.sqrt(mean_squared_error(all_targets, last_hour_mean_predictions))
    last_hour_mean_mae = mean_absolute_error(all_targets, last_hour_mean_predictions)
    last_hour_mean_r2 = r2_score(all_targets, last_hour_mean_predictions)
    
    baseline_results['last_hour_mean'] = {
        'rmse': last_hour_mean_rmse,
        'mae': last_hour_mean_mae,
        'r2': last_hour_mean_r2,
        'predictions': last_hour_mean_predictions
    }
    
    # Print baseline results
    print("\nBaseline Model Results:")
    print(f"Last Value - RMSE: {last_value_rmse:.4f}, MAE: {last_value_mae:.4f}, R²: {last_value_r2:.4f}")
    print(f"Linear Extrapolation - RMSE: {linear_extrapolation_rmse:.4f}, MAE: {linear_extrapolation_mae:.4f}, R²: {linear_extrapolation_r2:.4f}")
    print(f"Last Hour Mean - RMSE: {last_hour_mean_rmse:.4f}, MAE: {last_hour_mean_mae:.4f}, R²: {last_hour_mean_r2:.4f}")
    
    return baseline_results, all_targets

def compare_with_baselines(results, baseline_results, output_dir):
    """
    Compare the model performance with baseline models and visualize the results.
    """
    print("\nComparing model with baselines...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    model_metrics = {
        'rmse': results['rmse'],
        'mae': results['mae'],
        'r2': results['r2']
    }
    
    # Prepare data for bar charts
    models = ['Model', 'Last Value', 'Linear Extrapolation', 'Last Hour Mean']
    rmse_values = [
        model_metrics['rmse'],
        baseline_results['last_value']['rmse'],
        baseline_results['linear_extrapolation']['rmse'],
        baseline_results['last_hour_mean']['rmse']
    ]
    mae_values = [
        model_metrics['mae'],
        baseline_results['last_value']['mae'],
        baseline_results['linear_extrapolation']['mae'],
        baseline_results['last_hour_mean']['mae']
    ]
    r2_values = [
        model_metrics['r2'],
        baseline_results['last_value']['r2'],
        baseline_results['linear_extrapolation']['r2'],
        baseline_results['last_hour_mean']['r2']
    ]
    
    # Create bar charts
    plt.figure(figsize=(15, 5))
    
    # RMSE comparison
    plt.subplot(1, 3, 1)
    bars = plt.bar(models, rmse_values, color=['red', 'blue', 'green', 'orange'])
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # MAE comparison
    plt.subplot(1, 3, 2)
    bars = plt.bar(models, mae_values, color=['red', 'blue', 'green', 'orange'])
    plt.ylabel('MAE')
    plt.title('MAE Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # R² comparison
    plt.subplot(1, 3, 3)
    bars = plt.bar(models, r2_values, color=['red', 'blue', 'green', 'orange'])
    plt.ylabel('R²')
    plt.title('R² Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_vs_baselines.png'))
    print(f"Saved model vs baselines comparison to {os.path.join(output_dir, 'model_vs_baselines.png')}")
    
    # Create a table with all metrics
    metrics_table = pd.DataFrame({
        'Model': ['CAMIT-GF', 'Last Value', 'Linear Extrapolation', 'Last Hour Mean'],
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })
    
    # Save metrics table to CSV
    metrics_table.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"Saved metrics table to {os.path.join(output_dir, 'model_comparison.csv')}")
    
    # Determine the best model for each metric
    best_rmse_model = models[np.argmin(rmse_values)]
    best_mae_model = models[np.argmin(mae_values)]
    best_r2_model = models[np.argmax(r2_values)]
    
    print(f"\nBest model by RMSE: {best_rmse_model} ({min(rmse_values):.4f})")
    print(f"Best model by MAE: {best_mae_model} ({min(mae_values):.4f})")
    print(f"Best model by R²: {best_r2_model} ({max(r2_values):.4f})")
    
    # Calculate improvement/degradation compared to best baseline
    baseline_rmse_values = rmse_values[1:]  # Exclude the model
    baseline_mae_values = mae_values[1:]
    baseline_r2_values = r2_values[1:]
    
    best_baseline_rmse = min(baseline_rmse_values)
    best_baseline_mae = min(baseline_mae_values)
    
    rmse_improvement = (best_baseline_rmse - model_metrics['rmse']) / best_baseline_rmse * 100
    mae_improvement = (best_baseline_mae - model_metrics['mae']) / best_baseline_mae * 100
    
    # For R², improvement is different since higher is better
    if best_baseline_r2 <= 0 and model_metrics['r2'] <= 0:
        # Both are negative, less negative is better
        r2_improvement = (model_metrics['r2'] - best_baseline_r2) / abs(best_baseline_r2) * 100 if best_baseline_r2 != 0 else float('inf')
    elif best_baseline_r2 <= 0:
        # Baseline is negative, model is positive
        r2_improvement = float('inf')  # Infinite improvement
    elif model_metrics['r2'] <= 0:
        # Model is negative, baseline is positive
        r2_improvement = -float('inf')  # Infinite degradation
    else:
        # Both are positive, higher is better
        r2_improvement = (model_metrics['r2'] - best_baseline_r2) / best_baseline_r2 * 100
    
    print(f"\nCompared to best baseline:")
    print(f"RMSE: {'improvement' if rmse_improvement > 0 else 'degradation'} of {abs(rmse_improvement):.2f}%")
    print(f"MAE: {'improvement' if mae_improvement > 0 else 'degradation'} of {abs(mae_improvement):.2f}%")
    
    if r2_improvement == float('inf'):
        print(f"R²: infinite improvement (baseline was negative, model is positive)")
    elif r2_improvement == -float('inf'):
        print(f"R²: infinite degradation (baseline was positive, model is negative)")
    else:
        print(f"R²: {'improvement' if r2_improvement > 0 else 'degradation'} of {abs(r2_improvement):.2f}%")
    
    return {
        'model_metrics': model_metrics,
        'baseline_metrics': {
            'last_value': {k: v for k, v in baseline_results['last_value'].items() if k != 'predictions'},
            'linear_extrapolation': {k: v for k, v in baseline_results['linear_extrapolation'].items() if k != 'predictions'},
            'last_hour_mean': {k: v for k, v in baseline_results['last_hour_mean'].items() if k != 'predictions'}
        },
        'best_models': {
            'rmse': best_rmse_model,
            'mae': best_mae_model,
            'r2': best_r2_model
        },
        'improvements': {
            'rmse': rmse_improvement,
            'mae': mae_improvement,
            'r2': r2_improvement
        }
    }

def suggest_improvements(analysis_results, comparison_results, output_dir):
    """
    Suggest improvements to the model based on the analysis.
    """
    print("\n=== Model Improvement Suggestions ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    suggestions = []
    
    # Check if model is worse than baselines
    if comparison_results['improvements']['rmse'] < 0 or comparison_results['improvements']['mae'] < 0:
        suggestions.append("The model performs worse than simple baselines. Consider the following:")
        suggestions.append("1. Increase model capacity (more layers, larger d_model)")
        suggestions.append("2. Train for more epochs")
        suggestions.append("3. Use a different learning rate")
        suggestions.append("4. Add regularization to prevent overfitting")
    
    # Check for bias
    if abs(analysis_results['bias']) > 0.1:
        suggestions.append(f"The model has a bias of {analysis_results['bias']:.4f}. Consider:")
        suggestions.append("1. Check for data normalization issues")
        suggestions.append("2. Use a different loss function that penalizes bias")
        suggestions.append("3. Add more training data")
    
    # Check for correlation
    if analysis_results['correlation'] < 0.5:
        suggestions.append(f"The correlation between predictions and targets is low ({analysis_results['correlation']:.4f}). Consider:")
        suggestions.append("1. Add more features to the model")
        suggestions.append("2. Try a different model architecture")
        suggestions.append("3. Check for data quality issues")
    
    # Check for prediction range
    target_range = analysis_results['target_range']
    prediction_range = analysis_results['prediction_range']
    
    target_range_width = target_range[1] - target_range[0]
    prediction_range_width = prediction_range[1] - prediction_range[0]
    
    if prediction_range_width < 0.5 * target_range_width:
        suggestions.append("The model's prediction range is much narrower than the target range. Consider:")
        suggestions.append("1. Use a different loss function that penalizes range compression")
        suggestions.append("2. Add more diverse training data")
        suggestions.append("3. Check for gradient vanishing/exploding issues")
    
    # Check for prediction variance
    if analysis_results['prediction_std'] < 0.5 * analysis_results['target_std']:
        suggestions.append("The model's predictions have much lower variance than the targets. Consider:")
        suggestions.append("1. Add noise during training to increase prediction variance")
        suggestions.append("2. Use a different loss function that penalizes low variance")
        suggestions.append("3. Check for model collapse issues")
    
    # Specific suggestions for glucose prediction
    suggestions.append("\nSpecific suggestions for glucose prediction:")
    suggestions.append("1. Add more patient-specific features (age, weight, insulin sensitivity)")
    suggestions.append("2. Use a longer historical window (more than 4 hours)")
    suggestions.append("3. Add time-of-day features (circadian rhythms affect glucose)")
    suggestions.append("4. Consider using a recurrent architecture (LSTM/GRU) instead of or alongside the transformer")
    suggestions.append("5. Add meal composition features (fat, protein, not just carbs)")
    suggestions.append("6. Implement a multi-task learning approach (predict multiple future time points)")
    
    # Print and save suggestions
    for suggestion in suggestions:
        print(suggestion)
    
    with open(os.path.join(output_dir, 'improvement_suggestions.txt'), 'w') as f:
        f.write('\n'.join(suggestions))
    
    print(f"Saved improvement suggestions to {os.path.join(output_dir, 'improvement_suggestions.txt')}")
    print("=== End of Suggestions ===\n")
    
    return suggestions

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
    
    # Set device
    device = set_device(args.use_mps)
    
    # Determine optimal number of workers if not specified
    if args.num_workers is None:
        if device.type == 'mps':
            args.num_workers = 0  # MPS doesn't work well with multiple workers
        else:
            args.num_workers = min(4, multiprocessing.cpu_count())
    
    print(f"Using {args.num_workers} worker processes for data loading")
    
    # Load model
    try:
        model = load_model(args.checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Load test data
    try:
        print(f"Loading and preprocessing data from {args.data_file}")
        
        if args.fast_mode:
            print("Running in fast mode with limited data preprocessing")
            # In fast mode, we'll only load a subset of the data
            _, _, test_dataset = preprocess_data(args.data_file, fast_mode=True)
        else:
            _, _, test_dataset = preprocess_data(args.data_file)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda')
        )
        
        print(f"Test dataset size: {len(test_dataset)} samples")
    except Exception as e:
        print(f"Failed to load test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model
    print("Evaluating model on test data...")
    try:
        # Pass max_eval_batches to evaluate_model
        results = evaluate_model(model, test_loader, device, max_eval_batches=args.max_eval_batches)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print metrics
    print("\nTest Results:")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"MAPE: {results['mape']:.2f}%")
    
    # Print Clarke Error Grid results if available
    if 'zone_A_percent' in results:
        print("\nClarke Error Grid Analysis:")
        print(f"Zone A: {results['zone_A_percent']:.2f}% (Clinically Accurate)")
        print(f"Zone B: {results['zone_B_percent']:.2f}% (Benign Errors)")
        print(f"Zone C: {results['zone_C_percent']:.2f}% (Overcorrection)")
        print(f"Zone D: {results['zone_D_percent']:.2f}% (Dangerous Failure to Detect)")
        print(f"Zone E: {results['zone_E_percent']:.2f}% (Erroneous Treatment)")
    
    # Save metrics to file
    metrics = {
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'r2': float(results['r2']),
        'mape': float(results['mape'])
    }
    
    # Add Clarke Error Grid metrics if available
    if 'zone_A_percent' in results:
        metrics.update({
            'clarke_zone_A_percent': float(results['zone_A_percent']),
            'clarke_zone_B_percent': float(results['zone_B_percent']),
            'clarke_zone_C_percent': float(results['zone_C_percent']),
            'clarke_zone_D_percent': float(results['zone_D_percent']),
            'clarke_zone_E_percent': float(results['zone_E_percent'])
        })
    
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Saved metrics to {os.path.join(args.output_dir, 'test_metrics.json')}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'actual': results['targets'],
            'predicted': results['predictions'],
            'error': results['predictions'] - results['targets']
        })
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved all predictions to {predictions_path}")
    
    # Visualize predictions
    try:
        print(f"Visualizing {args.num_samples} sample predictions...")
        visualize_predictions(results, args.num_samples, args.output_dir)
    except Exception as e:
        print(f"Error visualizing predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Visualize sequences
    try:
        print("Visualizing 12 predicted sequences for the hour-ahead forecast...")
        visualize_sequences(results, args.output_dir)
    except Exception as e:
        print(f"Error visualizing sequences: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze model performance
    try:
        print("Analyzing model performance...")
        performance_analysis = analyze_model_performance(results, args.output_dir)
    except Exception as e:
        print(f"Error analyzing model performance: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate baseline models
    try:
        print("Evaluating baseline models...")
        baseline_results, all_targets = evaluate_baseline_models(test_loader, device, max_eval_batches=args.max_eval_batches)
    except Exception as e:
        print(f"Error evaluating baseline models: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare model with baselines
    try:
        print("Comparing model with baselines...")
        comparison_results = compare_with_baselines(results, baseline_results, args.output_dir)
    except Exception as e:
        print(f"Error comparing model with baselines: {e}")
        import traceback
        traceback.print_exc()
    
    # Suggest improvements
    try:
        print("Suggesting improvements...")
        improvements = suggest_improvements(performance_analysis, comparison_results, args.output_dir)
    except Exception as e:
        print(f"Error suggesting improvements: {e}")
        import traceback
        traceback.print_exc()
    
    print("Testing completed successfully!")

if __name__ == "__main__":
    main() 