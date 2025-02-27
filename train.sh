#!/bin/bash
#SBATCH --job-name=CAMIT-GF    # Job name
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-node=1    # Number of tasks per node
#SBATCH --cpus-per-task=8      # Number of CPU cores per task
#SBATCH --gres=gpu:4           # Number of GPUs (4 GPUs)
#SBATCH --mem=32G              # Memory per node
#SBATCH --time=72:00:00        # Time limit hrs:min:sec
#SBATCH --partition=Teach-Standard
#SBATCH --output=%x_%j.out     # Standard output log
#SBATCH --error=%x_%j.err      # Standard error log

# Get the number of GPUs from SLURM
export NUM_GPUS=$(echo $SLURM_GRES_DETAIL | grep -o 'gpu:[0-9]' | cut -d':' -f2)
echo "Number of GPUs allocated: $NUM_GPUS"

# Set environment variables
export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)

# PyTorch specific settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Set paths
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}

# Create and set scratch directories
SCRATCH_DIR=/disk/scratch/${STUDENT_ID}
mkdir -p ${SCRATCH_DIR}
export TMPDIR=${SCRATCH_DIR}
export TMP=${SCRATCH_DIR}

# Create output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=${SCRATCH_DIR}/checkpoints/${TIMESTAMP}
LOG_DIR=${SCRATCH_DIR}/logs/${TIMESTAMP}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# Activate conda environment
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Install/upgrade required packages
pip install --no-cache-dir torch==2.0.1 torchvision torchaudio scikit-learn pandas numpy tqdm

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Number of GPUs: $NUM_GPUS"
nvidia-smi

# Launch distributed training using torchrun
echo "Starting distributed training..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    train.py \
    --batch_size 32 \
    --grad_accum 8 \
    --epochs 100 \
    --lr 0.001 \
    --patience 7 \
    --d_model 64 \
    --nhead 8 \
    --num_encoder_layers 2 \
    --num_main_layers 2 \
    2>&1 | tee ${LOG_DIR}/training.log

# Copy results back to home directory
echo "Copying results to permanent storage..."
mkdir -p /home/${STUDENT_ID}/CAMIT-GF/checkpoints
mkdir -p /home/${STUDENT_ID}/CAMIT-GF/logs
cp -r ${CHECKPOINT_DIR} /home/${STUDENT_ID}/CAMIT-GF/checkpoints/
cp -r ${LOG_DIR} /home/${STUDENT_ID}/CAMIT-GF/logs/

echo "Training completed at: $(date)"
nvidia-smi 