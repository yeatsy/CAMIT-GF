#!/bin/sh
#SBATCH -N 1         # nodes requested
#SBATCH -n 1         # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=16000  # memory in Mb
#SBATCH --time=0-48:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

# Create scratch directory
mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Create directories for checkpoints and logs
mkdir -p ${TMP}/checkpoints/
mkdir -p ${TMP}/logs/

export CHECKPOINT_DIR=${TMP}/checkpoints/
export LOG_DIR=${TMP}/logs/

# Set environment variables for tqdm
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export TQDM_MININTERVAL=1.0

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Install required packages
pip install --no-cache-dir torch==2.0.1 torchvision torchaudio scikit-learn pandas numpy tqdm

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345

# Print GPU information
nvidia-smi

# Launch distributed training using torchrun
echo "Starting distributed training..."
torchrun --nproc_per_node=4 --standalone train.py \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --d_model 64 \
    --nhead 8 \
    --num_encoder_layers 2 \
    --num_main_layers 2

echo "Training completed. Check logs for training progress." 