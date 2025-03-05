# CAMIT-GF: Glucose Forecasting Model

This repository contains an optimized implementation of the CAMIT-GF (Continuous Attention-based Multi-Input Transformer for Glucose Forecasting) model, designed to predict future glucose values for patients with diabetes.

## Key Features

- **12-Step Sequence Prediction**: The model now directly predicts a sequence of 12 future glucose values (1 hour at 5-minute intervals)
- **Optimized Training**: Faster training with improved hyperparameters and MPS/CUDA optimizations
- **Enhanced Architecture**: Added future time embeddings and sequence decoder for better temporal modeling
- **Improved Loss Function**: Custom loss function with sequence consistency penalties

## Model Architecture

The CAMIT-GF model uses a transformer-based architecture with:

- Separate encoders for glucose, carbohydrates, bolus insulin, and basal insulin inputs
- Multi-cross attention mechanism to model interactions between different inputs
- LSTM layer for improved temporal modeling
- Future time embeddings to capture time-of-day patterns in predictions
- Sequence decoder to generate coherent sequences of future glucose values

## Usage

### Training

To train the model with optimized settings:

```bash
python train_mps.py --batch_size 128 --lr 0.001 --d_model 128 --num_encoder_layers 2 --num_main_layers 2 --dropout 0.1 --use_ema --warmup_steps 1000
```

Key parameters:
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--d_model`: Model dimension (default: 128)
- `--num_encoder_layers`: Number of encoder layers (default: 2)
- `--num_main_layers`: Number of main transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--use_ema`: Use exponential moving average of model weights
- `--warmup_steps`: Warmup steps for learning rate scheduler (default: 1000)
- `--use_amp`: Use automatic mixed precision (for CUDA only)

### Evaluation

To evaluate the model's performance on predicting 12 steps into the future:

```bash
python evaluate_model.py --checkpoint_path checkpoints/best_model.pt --output_dir evaluation_results --num_samples 10
```

Key parameters:
- `--checkpoint_path`: Path to the model checkpoint
- `--output_dir`: Directory to save evaluation results
- `--num_samples`: Number of sample predictions to visualize

## Performance Optimizations

The model has been optimized for faster training and better prediction accuracy:

1. **Architectural Improvements**:
   - Reduced number of layers for faster training
   - Added future time embeddings for better sequence prediction
   - Implemented sequence decoder for coherent predictions

2. **Training Optimizations**:
   - Increased batch size for better parallelism
   - Implemented learning rate scheduler with warmup
   - Added exponential moving average (EMA) for more stable training
   - Optimized data loading with persistent workers

3. **Hardware Optimizations**:
   - MPS support for Apple Silicon
   - CUDA optimizations with TF32 support
   - Automatic mixed precision (AMP) for faster training on GPUs

## Results

The model achieves improved accuracy in predicting future glucose values, with particular emphasis on maintaining prediction quality across all 12 future time steps. The evaluation script generates detailed metrics and visualizations to assess model performance.

## Requirements

- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## License

[MIT License](LICENSE)
