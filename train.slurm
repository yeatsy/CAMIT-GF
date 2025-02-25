#!/bin/bash
#SBATCH --job-name=camit_gf_train    # Job name
#SBATCH --output=logs/slurm_%j.out   # Output file (%j expands to jobId)
#SBATCH --error=logs/slurm_%j.err    # Error file (%j expands to jobId)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=32G                    # Memory per node
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --partition=gpu              # Request GPU partition/queue

# Load necessary modules (modify these according to your cluster's setup)
module purge
module load cuda/11.8
module load python/3.9

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set up logging directory
mkdir -p logs

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"

# Print GPU information
nvidia-smi

# Run the training script
python train.py

# Print completion time
echo "Finished at: $(date)" 