import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define constants
SEQ_LEN = 48  # 4 hours at 5-minute intervals (4 * 60 / 5 = 48)
FORECAST_HORIZON = 12  # 1 hour ahead prediction (60 / 5 = 12)
BATCH_SIZE = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def preprocess_data(file_path, fast_mode=False):
    """
    Preprocess glucose data, splitting by patients for training, validation, and testing.
    Uses a 4-hour lookback window (48 time steps at 5-minute intervals) to predict glucose 1 hour ahead.
    
    Args:
        file_path (str): Path to the CSV file with patient data.
        fast_mode (bool): If True, only process a small subset of data for faster testing.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load and sort the data
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')
    df = df.sort_values(by=['id', 'time'])

    # Get unique patient IDs
    patient_ids = df['id'].unique()
    
    # In fast mode, only use a small subset of patients
    if fast_mode:
        max_patients = min(5, len(patient_ids))
        print(f"Fast mode: Using only {max_patients} patients for testing")
        patient_ids = patient_ids[:max_patients]

    # Split patient IDs into train, val, and test sets
    train_ids, temp_ids = train_test_split(patient_ids, train_size=TRAIN_RATIO, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42)
    
    # In fast mode, limit the number of sequences per patient
    max_sequences_per_patient = 100 if fast_mode else float('inf')

    # Helper function to create sequences for a set of patient IDs
    def create_sequences_for_ids(ids):
        data_list = []
        for patient_id in ids:
            group = df[df['id'] == patient_id]
            data = group[['glucose', 'basal', 'bolus', 'carbs']].values
            
            # Store original glucose values before standardization
            original_glucose = data[:, 0].copy()
            
            # Standardize data per patient
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            stds = np.where(stds == 0, 1e-8, stds)  # Avoid division by zero
            standardized_data = (data - means) / stds
            
            # Create sequences with 4-hour lookback (SEQ_LEN=48) for 1-hour prediction (FORECAST_HORIZON=12)
            num_samples = len(standardized_data) - SEQ_LEN - FORECAST_HORIZON + 1
            
            # In fast mode, limit the number of sequences
            if fast_mode:
                num_samples = min(num_samples, max_sequences_per_patient)
            
            for i in range(num_samples):
                # Extract 4 hours of input data (48 time steps)
                glucose_seq = standardized_data[i:i + SEQ_LEN, 0]
                basal_seq = standardized_data[i:i + SEQ_LEN, 1]
                bolus_seq = standardized_data[i:i + SEQ_LEN, 2]
                carbs_seq = standardized_data[i:i + SEQ_LEN, 3]
                
                # Target is next 12 timesteps (1 hour ahead)
                target = standardized_data[i + SEQ_LEN:i + SEQ_LEN + FORECAST_HORIZON, 0]
                
                # Get original glucose values for visualization
                original_glucose_seq = original_glucose[i:i + SEQ_LEN]
                original_target = original_glucose[i + SEQ_LEN:i + SEQ_LEN + FORECAST_HORIZON]
                
                # Store standardization parameters for this sequence
                glucose_mean = means[0]
                glucose_std = stds[0]
                
                # Skip sequences with NaN values
                if (np.isnan(glucose_seq).any() or np.isnan(basal_seq).any() or 
                    np.isnan(bolus_seq).any() or np.isnan(carbs_seq).any() or 
                    np.isnan(target).any()):
                    continue
                    
                data_list.append((
                    glucose_seq, basal_seq, bolus_seq, carbs_seq, target,
                    original_glucose_seq, original_target, glucose_mean, glucose_std
                ))
                
                # In fast mode, break early after collecting enough sequences
                if fast_mode and len(data_list) >= max_sequences_per_patient * len(ids):
                    break
                    
        return data_list

    # Create datasets for each split
    if fast_mode:
        # In fast mode, we only need the test dataset
        train_data = []
        val_data = []
        test_data = create_sequences_for_ids(test_ids)
    else:
        train_data = create_sequences_for_ids(train_ids)
        val_data = create_sequences_for_ids(val_ids)
        test_data = create_sequences_for_ids(test_ids)
    
    print(f"Created {len(train_data)} training sequences")
    print(f"Created {len(val_data)} validation sequences")
    print(f"Created {len(test_data)} test sequences")
    print(f"Each sequence uses {SEQ_LEN} time steps (4 hours) of data to predict glucose {FORECAST_HORIZON} time steps (1 hour) ahead")

    # In fast mode, use the simpler dataset class for faster loading
    if fast_mode:
        return (
            GlucoseDataset(train_data), 
            GlucoseDataset(val_data), 
            GlucoseDataset(test_data)
        )
    else:
        return (
            OptimizedGlucoseDataset(train_data), 
            OptimizedGlucoseDataset(val_data), 
            OptimizedGlucoseDataset(test_data)
        )

class GlucoseDataset(Dataset):
    """Custom Dataset for glucose time series."""
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        glucose_seq, basal_seq, bolus_seq, carbs_seq, target, original_glucose_seq, original_target, glucose_mean, glucose_std = self.data_list[idx]
        return {
            'glucose': torch.tensor(glucose_seq, dtype=torch.float32).unsqueeze(-1),
            'basal': torch.tensor(basal_seq, dtype=torch.float32).unsqueeze(-1),
            'bolus': torch.tensor(bolus_seq, dtype=torch.float32).unsqueeze(-1),
            'carbs': torch.tensor(carbs_seq, dtype=torch.float32).unsqueeze(-1),
            'target': torch.tensor(target, dtype=torch.float32),  # Shape will be [FORECAST_HORIZON]
            'original_glucose': torch.tensor(original_glucose_seq, dtype=torch.float32).unsqueeze(-1),
            'original_target': torch.tensor(original_target, dtype=torch.float32),  # Shape will be [FORECAST_HORIZON]
            'glucose_mean': torch.tensor(glucose_mean, dtype=torch.float32),
            'glucose_std': torch.tensor(glucose_std, dtype=torch.float32)
        }

class OptimizedGlucoseDataset(Dataset):
    """Optimized Dataset for glucose time series with pre-converted tensors."""
    def __init__(self, data_list):
        # Pre-convert all data to tensors for faster loading
        print("Pre-converting data to tensors for faster loading...")
        
        # Allocate memory for all tensors at once
        n_samples = len(data_list)
        self.glucose = torch.zeros((n_samples, SEQ_LEN, 1), dtype=torch.float32)
        self.basal = torch.zeros((n_samples, SEQ_LEN, 1), dtype=torch.float32)
        self.bolus = torch.zeros((n_samples, SEQ_LEN, 1), dtype=torch.float32)
        self.carbs = torch.zeros((n_samples, SEQ_LEN, 1), dtype=torch.float32)
        self.targets = torch.zeros((n_samples, FORECAST_HORIZON), dtype=torch.float32)  # Changed to store 12 timesteps
        self.original_glucose = torch.zeros((n_samples, SEQ_LEN, 1), dtype=torch.float32)
        self.original_targets = torch.zeros((n_samples, FORECAST_HORIZON), dtype=torch.float32)  # Changed to store 12 timesteps
        self.glucose_means = torch.zeros(n_samples, dtype=torch.float32)
        self.glucose_stds = torch.zeros(n_samples, dtype=torch.float32)
        
        # Fill tensors with data
        for i, (glucose_seq, basal_seq, bolus_seq, carbs_seq, target, original_glucose_seq, original_target, glucose_mean, glucose_std) in enumerate(data_list):
            self.glucose[i, :, 0] = torch.tensor(glucose_seq, dtype=torch.float32)
            self.basal[i, :, 0] = torch.tensor(basal_seq, dtype=torch.float32)
            self.bolus[i, :, 0] = torch.tensor(bolus_seq, dtype=torch.float32)
            self.carbs[i, :, 0] = torch.tensor(carbs_seq, dtype=torch.float32)
            self.targets[i] = torch.tensor(target, dtype=torch.float32)  # Now stores all 12 timesteps
            self.original_glucose[i, :, 0] = torch.tensor(original_glucose_seq, dtype=torch.float32)
            self.original_targets[i] = torch.tensor(original_target, dtype=torch.float32)  # Now stores all 12 timesteps
            self.glucose_means[i] = torch.tensor(glucose_mean, dtype=torch.float32)
            self.glucose_stds[i] = torch.tensor(glucose_std, dtype=torch.float32)
        
        print(f"Converted {n_samples} sequences to tensors")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'glucose': self.glucose[idx],
            'basal': self.basal[idx],
            'bolus': self.bolus[idx],
            'carbs': self.carbs[idx],
            'target': self.targets[idx],
            'original_glucose': self.original_glucose[idx],
            'original_target': self.original_targets[idx],
            'glucose_mean': self.glucose_means[idx],
            'glucose_std': self.glucose_stds[idx]
        }

# Example usage
if __name__ == "__main__":
    file_path = 'full_patient_dataset.csv'
    train_dataset, val_dataset, test_dataset = preprocess_data(file_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Test a batch
    for batch in test_loader:
        print("Glucose shape:", batch['glucose'].shape)  # (batch_size, SEQ_LEN, 1)
        print("Target shape:", batch['target'].shape)    # (batch_size, FORECAST_HORIZON)
        break