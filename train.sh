#!/bin/sh
#SBATCH -N 1         # nodes requested
#SBATCH -n 1         # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=32000  # memory in Mb - increased for larger batch sizes
#SBATCH --time=0-72:00:00  # increased time for hyperparameter search

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

# Set PyTorch memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8

# Enable CUDA memory stats for debugging
export CUDA_MEMORY_DEBUG=1

# Set environment variables for better GPU utilization
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Create directories for data and outputs
mkdir -p ${TMP}/datasets/
mkdir -p ${TMP}/checkpoints/
mkdir -p ${TMP}/logs/

export DATASET_DIR=${TMP}/datasets/
export CHECKPOINT_DIR=${TMP}/checkpoints/
export LOG_DIR=${TMP}/logs/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Install required packages
pip install --no-cache-dir torch torchvision torchaudio scikit-learn tqdm pandas numpy

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"

# Print GPU information
nvidia-smi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ${CHECKPOINT_DIR}/${TIMESTAMP}
mkdir -p ${LOG_DIR}/${TIMESTAMP}

# Set up logging
exec 1>${LOG_DIR}/${TIMESTAMP}/stdout.log
exec 2>${LOG_DIR}/${TIMESTAMP}/stderr.log

echo "Starting hyperparameter search and training..."
echo "Logs will be saved to: ${LOG_DIR}/${TIMESTAMP}"
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}/${TIMESTAMP}"

# Run the training script
python model.py \
    --checkpoint_dir ${CHECKPOINT_DIR}/${TIMESTAMP} \
    --log_dir ${LOG_DIR}/${TIMESTAMP} \
    2>&1 | tee ${LOG_DIR}/${TIMESTAMP}/training.log

# Copy results to permanent storage
echo "Copying results to permanent storage..."
cp -r ${CHECKPOINT_DIR}/${TIMESTAMP} /home/${STUDENT_ID}/CAMIT-GF/checkpoints/
cp -r ${LOG_DIR}/${TIMESTAMP} /home/${STUDENT_ID}/CAMIT-GF/logs/

# Print completion information
echo "Training completed"
echo "Finished at: $(date)"

# Print final GPU status
nvidia-smi 