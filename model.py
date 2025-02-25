import torch
import torch.nn as nn
import math

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
    def __init__(self, d_model, nhead):
        super().__init__()
        # Attention layers with batch_first=True for MPS compatibility
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_carbs = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_bolus = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_basal = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        # Layer normalization for each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)

    def forward(self, glucose_hidden, carbs_hidden, bolus_hidden, basal_hidden):
        # Self-attention on glucose sequence
        attn_output, _ = self.self_attn(glucose_hidden, glucose_hidden, glucose_hidden)
        glucose_hidden = self.norm1(glucose_hidden + attn_output)

        # Cross-attention to carbs sequence
        attn_output, _ = self.cross_attn_carbs(glucose_hidden, carbs_hidden, carbs_hidden)
        glucose_hidden = self.norm2(glucose_hidden + attn_output)

        # Cross-attention to bolus sequence
        attn_output, _ = self.cross_attn_bolus(glucose_hidden, bolus_hidden, bolus_hidden)
        glucose_hidden = self.norm3(glucose_hidden + attn_output)

        # Cross-attention to basal sequence
        attn_output, _ = self.cross_attn_basal(glucose_hidden, basal_hidden, basal_hidden)
        glucose_hidden = self.norm4(glucose_hidden + attn_output)

        # Feedforward network
        ff_output = self.feed_forward(glucose_hidden)
        glucose_hidden = self.norm5(glucose_hidden + ff_output)

        return glucose_hidden

# Main CAMIT-GF model class
class CAMIT_GF(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_main_layers):
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
            MultiCrossAttentionTransformerLayer(d_model, nhead) for _ in range(num_main_layers)
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
    # Check for MPS availability for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    seq_len = 288  # 24 hours at 5-min intervals
    d_model = 64
    nhead = 8
    num_encoder_layers = 2
    num_main_layers = 2

    # Initialize model and move to device
    model = CAMIT_GF(d_model, nhead, num_encoder_layers, num_main_layers).to(device)

    # Generate dummy data for demonstration
    glucose = torch.randn(batch_size, seq_len, 1).to(device)
    carbs = torch.randn(batch_size, seq_len, 1).to(device)
    bolus = torch.randn(batch_size, seq_len, 1).to(device)
    basal = torch.randn(batch_size, seq_len, 1).to(device)
    target = torch.randn(batch_size, 1).to(device)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example training step
    model.train()
    optimizer.zero_grad()
    prediction = model(glucose, carbs, bolus, basal)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

    print("Training step completed.")
    print(f"Loss: {loss.item():.4f}")