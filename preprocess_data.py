import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define constants
SEQ_LEN = 288  # 24 hours at 5-minute intervals (24 * 60 / 5)
FORECAST_HORIZON = 12  # 60 minutes ahead (60 / 5)
BATCH_SIZE = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def preprocess_data(file_path):
    """
    Preprocess glucose data, splitting by patients for training, validation, and testing.
    
    Args:
        file_path (str): Path to the CSV file with patient data.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load and sort the data
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')
    df = df.sort_values(by=['id', 'time'])

    # Get unique patient IDs
    patient_ids = df['id'].unique()

    # Split patient IDs into train, val, and test sets
    train_ids, temp_ids = train_test_split(patient_ids, train_size=TRAIN_RATIO, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42)

    # Helper function to create sequences for a set of patient IDs
    def create_sequences_for_ids(ids):
        data_list = []
        for patient_id in ids:
            group = df[df['id'] == patient_id]
            data = group[['glucose', 'basal', 'bolus', 'carbs']].values
            
            # Standardize data per patient
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            stds = np.where(stds == 0, 1e-8, stds)  # Avoid division by zero
            standardized_data = (data - means) / stds
            
            # Create sequences
            num_samples = len(standardized_data) - SEQ_LEN - FORECAST_HORIZON + 1
            for i in range(num_samples):
                glucose_seq = standardized_data[i:i + SEQ_LEN, 0]
                basal_seq = standardized_data[i:i + SEQ_LEN, 1]
                bolus_seq = standardized_data[i:i + SEQ_LEN, 2]
                carbs_seq = standardized_data[i:i + SEQ_LEN, 3]
                target = standardized_data[i + SEQ_LEN + FORECAST_HORIZON - 1, 0]
                data_list.append((glucose_seq, basal_seq, bolus_seq, carbs_seq, target))
        return data_list

    # Create datasets for each split
    train_data = create_sequences_for_ids(train_ids)
    val_data = create_sequences_for_ids(val_ids)
    test_data = create_sequences_for_ids(test_ids)

    return GlucoseDataset(train_data), GlucoseDataset(val_data), GlucoseDataset(test_data)

class GlucoseDataset(Dataset):
    """Custom Dataset for glucose time series."""
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        glucose_seq, basal_seq, bolus_seq, carbs_seq, target = self.data_list[idx]
        return {
            'glucose': torch.tensor(glucose_seq, dtype=torch.float32).unsqueeze(-1),
            'basal': torch.tensor(basal_seq, dtype=torch.float32).unsqueeze(-1),
            'bolus': torch.tensor(bolus_seq, dtype=torch.float32).unsqueeze(-1),
            'carbs': torch.tensor(carbs_seq, dtype=torch.float32).unsqueeze(-1),
            'target': torch.tensor(target, dtype=torch.float32)
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
        print("Target shape:", batch['target'].shape)    # (batch_size,)
        break