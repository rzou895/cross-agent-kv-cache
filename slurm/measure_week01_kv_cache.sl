#!/bin/bash -e

#SBATCH --job-name=measure_kv_cache
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/week01/measure_kv_cache_%j.out
#SBATCH --error=results/week01/measure_kv_cache_%j.err

cd /nesi/project/uoa04658/rzou895/cross-agent-kv-cache

echo "===== Initialising environment ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

module purge
module load Miniforge3/25.3.1-0

export PYTHONNOUSERSITE=1

export HF_HOME=/nesi/nobackup/uoa04658/rzou895/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ENV=/nesi/project/uoa04658/rzou895/conda_envs/cross-agent-kv-cache
PYTHON="${ENV}/bin/python"

echo "===== Checking GPU ====="

"${PYTHON}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== Measuring KV cache ====="

srun "${PYTHON}" -m analysis.measure_week01_kv_cache

echo "===== KV cache measurement completed ====="
