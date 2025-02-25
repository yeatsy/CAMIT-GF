#!/bin/sh
#SBATCH -N 1         # nodes requested
#SBATCH -n 1         # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=16000    # memory in Mb
#SBATCH --time=0-48:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Install required packages if they're not already installed
pip install --no-cache-dir scikit-learn tqdm pandas numpy

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"

# Print GPU information
nvidia-smi

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=4

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

# Launch distributed training using torchrun
echo "Starting distributed training with 4 GPUs..."
torchrun --nproc_per_node=4 --standalone train.py \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.001

# Print completion time
echo "Finished at: $(date)" 