#!/bin/bash -e

#SBATCH --job-name=qwen_smoke
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=qwen_smoke_%j.out
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

cd /nesi/project/uoa04658/rzou895/cross-agent-kv-cache

echo "===== Slurm information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

echo
echo "===== Loaded modules ====="
module list

echo
echo "===== NVIDIA information ====="
nvidia-smi

echo
echo "===== Starting Python test ====="

srun --ntasks=1 \
    python -u scripts/test_qwen_generation.py