#!/bin/bash -e

#SBATCH --job-name=benchmark_day5
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/week01/benchmark_day5_%j.out
#SBATCH --error=results/week01/benchmark_day5_%j.err

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

SEED="${SEED:-1234}"
OUTPUT="${OUTPUT:-results/week01/single_model_fp16_reproduce.jsonl}"
SMOKE="${SMOKE:-0}"

echo "===== Benchmark configuration ====="
echo "Python: ${PYTHON}"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT}"
echo "Smoke mode: ${SMOKE}"
echo "HF_HOME: ${HF_HOME}"

"${PYTHON}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

MODEL_CACHE="${HF_HUB_CACHE}/models--Qwen--Qwen2.5-0.5B-Instruct"

if [[ ! -d "${MODEL_CACHE}" ]]; then
    echo "ERROR: Cached model was not found:"
    echo "${MODEL_CACHE}"
    exit 1
fi

echo "===== GPU information ====="
nvidia-smi

ARGS=(
    --seed "${SEED}"
    --output "${OUTPUT}"
)

if [[ "${SMOKE}" == "1" ]]; then
    ARGS+=(--smoke)
fi

echo "===== Running benchmark ====="

srun "${PYTHON}" -m harness.benchmark_single_model "${ARGS[@]}"

echo "===== Benchmark completed ====="
