#!/bin/bash -e

#SBATCH --job-name=inspect_kv_cache
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/inspect_kv_cache_%j.out
#SBATCH --error=results/inspect_kv_cache_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rzou895@aucklanduni.ac.nz

echo "===== Initialising environment ====="

export PYTHONNOUSERSITE=1

ENV_DIR=/nesi/project/uoa04658/rzou895/conda_envs/cross-agent-kv-cache
ENV_PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "${ENV_PYTHON}" ]]; then
    echo "ERROR: Python executable not found: ${ENV_PYTHON}" >&2
    exit 1
fi

export HF_HOME=/nesi/nobackup/uoa04658/rzou895/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME=/nesi/nobackup/uoa04658/rzou895/torch
export XDG_CACHE_HOME=/nesi/nobackup/uoa04658/rzou895/cache

mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${XDG_CACHE_HOME}"

cd "${SLURM_SUBMIT_DIR}"

echo "===== Slurm information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: ${ENV_PYTHON}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

echo "===== Checking Python environment ====="
"${ENV_PYTHON}" -c \
'import sys, torch; print("Python:", sys.executable); print("Torch:", torch.__version__)'

echo "===== GPU information ====="
nvidia-smi

echo "===== Inspecting real KV cache ====="
"${ENV_PYTHON}" -u harness/inspect_kv_cache.py

echo "===== Running lightweight tests ====="
"${ENV_PYTHON}" -u tests/test_kv_shape.py

echo "===== Job completed ====="