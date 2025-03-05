import unittest
import torch
import numpy as np
from model import CAMIT_GF, PositionalEncoding, MultiCrossAttentionTransformerLayer
from preprocess_data import FORECAST_HORIZON

class TestCAMIT_GF(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Match default model configuration
        self.batch_size = 4
        self.seq_length = 24
        self.d_model = 64
        self.nhead = 4
        self.num_encoder_layers = 2
        self.dropout = 0.5  # Increased dropout for more visible effect
        
        # Initialize model with default configuration
        self.model = CAMIT_GF(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Create sample input data matching the model's expected shapes
        torch.manual_seed(42)  # For reproducibility
        self.glucose = torch.randn(self.batch_size, self.seq_length, 1).to(self.device)
        self.carbs = torch.randn(self.batch_size, self.seq_length, 1).to(self.device)
        self.bolus = torch.randn(self.batch_size, self.seq_length, 1).to(self.device)
        self.basal = torch.randn(self.batch_size, self.seq_length, 1).to(self.device)

    def test_model_initialization(self):
        """Test if model initializes correctly with given parameters"""
        # Test model structure and components
        self.assertIsInstance(self.model.encoder, torch.nn.TransformerEncoder)
        self.assertEqual(len(self.model.encoder.layers), self.num_encoder_layers)
        self.assertIsInstance(self.model.pos_encoding, PositionalEncoding)
        
        # Test output shape
        with torch.no_grad():
            output = self.model(self.glucose, self.carbs, self.bolus, self.basal)
            self.assertEqual(output.shape, (self.batch_size, FORECAST_HORIZON))

    def test_training_step(self):
        """Test if model can perform a training step without errors"""
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Forward pass
        output1 = self.model(self.glucose, self.carbs, self.bolus, self.basal)
        
        # Create dummy target with significant difference from output
        target = output1.clone() + 1.0
        
        # Calculate loss
        loss = torch.nn.MSELoss()(output1, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Forward pass after optimization
        output2 = self.model(self.glucose, self.carbs, self.bolus, self.basal)
        
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.item() > 0)  # Loss should be positive
        # Outputs should be different after optimization step
        self.assertFalse(torch.allclose(output1, output2, rtol=1e-3, atol=1e-3))

    def test_input_validation(self):
        """Test if model handles invalid inputs appropriately"""
        # Test with mismatched sequence lengths between inputs
        invalid_glucose = torch.randn(self.batch_size, self.seq_length + 1, 1).to(self.device)
        invalid_carbs = torch.randn(self.batch_size, self.seq_length - 1, 1).to(self.device)
        
        with self.assertRaises(Exception):  # Either RuntimeError or ValueError
            self.model(invalid_glucose, self.carbs, self.bolus, self.basal)

    def test_positional_encoding(self):
        """Test if positional encoding works correctly"""
        pos_encoder = PositionalEncoding(self.d_model).to(self.device)
        x = torch.zeros(self.batch_size, self.seq_length, self.d_model).to(self.device)
        encoded = pos_encoder(x)
        
        # Check output shape
        self.assertEqual(encoded.shape, (self.batch_size, self.seq_length, self.d_model))
        # Check if encoding is added to input
        self.assertFalse(torch.allclose(encoded, x))
        # Check if output is on the correct device
        self.assertEqual(encoded.device.type, self.device.type)

    def test_transformer_layer(self):
        """Test if transformer layer processes inputs correctly"""
        layer = MultiCrossAttentionTransformerLayer(self.d_model, self.nhead).to(self.device)
        
        # Create sample hidden states
        glucose_hidden = torch.randn(self.batch_size, self.seq_length, self.d_model).to(self.device)
        carbs_hidden = torch.randn(self.batch_size, self.seq_length, self.d_model).to(self.device)
        bolus_hidden = torch.randn(self.batch_size, self.seq_length, self.d_model).to(self.device)
        basal_hidden = torch.randn(self.batch_size, self.seq_length, self.d_model).to(self.device)
        
        output = layer(glucose_hidden, carbs_hidden, bolus_hidden, basal_hidden)
        
        # Check output shape
        self.assertEqual(output.shape, glucose_hidden.shape)
        # Check if output is on the correct device
        self.assertEqual(output.device.type, glucose_hidden.device.type)

    def test_forward_pass(self):
        """Test if forward pass works with different input configurations"""
        torch.manual_seed(42)  # For reproducibility
        
        # Create inputs with significantly different last glucose values
        glucose1 = self.glucose.clone()
        glucose2 = self.glucose.clone()
        glucose2[:, -1, :] = glucose1[:, -1, :] + 2.0  # Add significant difference to last value
        
        # Test with different glucose values but same time indices
        output1 = self.model(glucose1, self.carbs, self.bolus, self.basal)
        output2 = self.model(glucose2, self.carbs, self.bolus, self.basal)
        
        # Outputs should be different due to different last glucose values
        self.assertFalse(torch.allclose(output1, output2, rtol=1e-3, atol=1e-3))
        
        # Test shape and device
        self.assertEqual(output1.shape, (self.batch_size, FORECAST_HORIZON))
        self.assertEqual(output1.device.type, self.device.type)
        
        # Verify the residual connection's effect
        # The difference in outputs should be correlated with the difference in last glucose values
        output_diff = torch.abs(output2 - output1).mean()
        glucose_diff = torch.abs(glucose2[:, -1, :] - glucose1[:, -1, :]).mean()
        self.assertGreater(output_diff, glucose_diff * 0.1)  # Output difference should be significant

    def test_model_evaluation(self):
        """Test if model can switch between train and eval modes"""
        torch.manual_seed(42)  # For reproducibility
        
        # Test eval mode - outputs should be deterministic
        self.model.eval()
        with torch.no_grad():
            eval_output1 = self.model(self.glucose, self.carbs, self.bolus, self.basal)
            eval_output2 = self.model(self.glucose, self.carbs, self.bolus, self.basal)
        
        # In eval mode, outputs should be identical
        self.assertTrue(torch.allclose(eval_output1, eval_output2, rtol=1e-5, atol=1e-5))
        
        # Test training mode - outputs should vary due to dropout
        self.model.train()
        
        # Create inputs with more significant features to make dropout effects more visible
        glucose = torch.randn(self.batch_size, self.seq_length, 1).to(self.device) * 2.0
        carbs = torch.randn(self.batch_size, self.seq_length, 1).to(self.device) * 2.0
        bolus = torch.randn(self.batch_size, self.seq_length, 1).to(self.device) * 2.0
        basal = torch.randn(self.batch_size, self.seq_length, 1).to(self.device) * 2.0
        
        # Run multiple forward passes and collect outputs
        train_outputs = []
        for _ in range(5):
            out = self.model(glucose, carbs, bolus, basal)
            train_outputs.append(out)
        
        # Calculate maximum difference between any two outputs
        max_diff = 0
        for i in range(len(train_outputs)):
            for j in range(i + 1, len(train_outputs)):
                diff = torch.max(torch.abs(train_outputs[i] - train_outputs[j])).item()
                max_diff = max(max_diff, diff)
        
        # The maximum difference should be significant due to dropout
        self.assertGreater(max_diff, 1e-2, f"Dropout should cause variations in training mode (max diff: {max_diff})")

if __name__ == '__main__':
    unittest.main() 