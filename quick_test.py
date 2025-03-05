import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from model import CAMIT_GF
from preprocess_data import preprocess_data, FORECAST_HORIZON
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description='Quick test for CAMIT-GF Model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--data_file', type=str, default='full_patient_dataset.csv',
                      help='Path to the test data')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of samples to visualize')
    parser.add_argument('--use_mps', action='store_true',
                      help='Use MPS (Metal Performance Shaders) on Apple Silicon')
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
                'dropout': 0.0  # No dropout during testing
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

def quick_test(model, test_loader, device, num_samples=5):
    """Run a quick test on the model."""
    model.eval()
    all_predictions = []
    all_targets = []
    visualization_data = []
    
    print("\nRunning quick test...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            glucose = batch['glucose'].to(device)
            carbs = batch['carbs'].to(device)
            bolus = batch['bolus'].to(device)
            basal = batch['basal'].to(device)
            targets = batch['target']
            
            # Get predictions
            outputs = model(glucose, carbs, bolus, basal)
            
            # Store results
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
            
            # Store some samples for visualization
            if len(visualization_data) < num_samples:
                for i in range(min(num_samples - len(visualization_data), len(outputs))):
                    visualization_data.append({
                        'input': glucose[i].cpu().numpy(),
                        'prediction': outputs[i].cpu().numpy(),
                        'target': targets[i].numpy()
                    })
            
            if len(visualization_data) >= num_samples:
                break
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics, visualization_data

def visualize_results(visualization_data):
    """Create plots of predictions vs actual values."""
    plt.figure(figsize=(15, 5))
    
    # Plot predictions for each sample
    for i, data in enumerate(visualization_data):
        plt.subplot(1, len(visualization_data), i + 1)
        
        # Plot input sequence
        plt.plot(data['input'], label='Input', color='blue', alpha=0.5)
        
        # Plot prediction and target
        x_pred = np.arange(len(data['input']), len(data['input']) + FORECAST_HORIZON)
        plt.plot(x_pred, data['prediction'], label='Prediction', color='red')
        plt.plot(x_pred, data['target'], label='Target', color='green')
        
        plt.title(f'Sample {i+1}')
        if i == 0:
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png')
    plt.close()

def main():
    args = parse_args()
    device = set_device(args.use_mps)
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, _, test_dataset = preprocess_data(args.data_file)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    model = load_model(args.checkpoint_path, device)
    
    # Run quick test
    metrics, visualization_data = quick_test(model, test_loader, device, args.num_samples)
    
    # Print metrics
    print("\nTest Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(visualization_data)
    print("Results visualization saved as 'quick_test_results.png'")

if __name__ == "__main__":
    main() 