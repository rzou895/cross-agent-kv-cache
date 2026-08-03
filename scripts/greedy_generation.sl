#!/bin/bash -e

#SBATCH --job-name=greedy_generation
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/greedy_generation_%j.out
#SBATCH --error=results/greedy_generation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rzou895@aucklanduni.ac.nz

echo "===== Initialising environment ====="

module purge
module load Miniforge3/25.3.1-0

source "$(conda info --base)/etc/profile.d/conda.sh"

export PYTHONNOUSERSITE=1

conda activate \
/nesi/project/uoa04658/rzou895/conda_envs/cross-agent-kv-cache

# Store downloadable caches in Nobackup.
export HF_HOME=/nesi/nobackup/uoa04658/rzou895/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME=/nesi/nobackup/uoa04658/rzou895/torch
export XDG_CACHE_HOME=/nesi/nobackup/uoa04658/rzou895/cache

mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${XDG_CACHE_HOME}"

# Enter the directory from which sbatch was submitted.
cd "${SLURM_SUBMIT_DIR}"

echo "===== Slurm information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

echo "===== GPU information ====="
nvidia-smi

echo "===== Python environment ====="
python -u scripts/check_environment.py

echo "===== Greedy generation ====="
python -u scripts/test_generation.py

echo "===== Job completed ====="