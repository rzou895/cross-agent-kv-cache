#!/bin/bash -e

#SBATCH --job-name=benchmark_smoke
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/week01/benchmark_smoke_%j.out
#SBATCH --error=results/week01/benchmark_smoke_%j.err

cd /nesi/project/uoa04658/rzou895/cross-agent-kv-cache

echo "===== Initialising environment ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

module purge
module load Miniforge3

# Use the Hugging Face cache created by previous jobs.
export HF_HOME=/nesi/nobackup/uoa04658/rzou895/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"

# The model has already been downloaded, so use the local cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${HF_HUB_CACHE}"

ENV=/nesi/project/uoa04658/rzou895/conda_envs/cross-agent-kv-cache
PYTHON="${ENV}/bin/python"

echo "===== Checking environment ====="
echo "Python: ${PYTHON}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"

"${PYTHON}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== Checking cached model ====="

MODEL_CACHE="${HF_HUB_CACHE}/models--Qwen--Qwen2.5-0.5B-Instruct"

if [[ ! -d "${MODEL_CACHE}" ]]; then
    echo "ERROR: Cached model was not found at:"
    echo "${MODEL_CACHE}"
    exit 1
fi

echo "Model cache found: ${MODEL_CACHE}"

echo "===== GPU information ====="
nvidia-smi

echo "===== Running benchmark smoke test ====="

srun "${PYTHON}" harness/benchmark_single_model.py --smoke

echo "===== Smoke test completed ====="
