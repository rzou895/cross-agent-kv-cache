#!/bin/bash -e

#SBATCH --job-name=collect_kv_calib
#SBATCH --account=uoa04658
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/week02/collect_kv_calib_%j.out
#SBATCH --error=results/week02/collect_kv_calib_%j.err

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
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Both the model and Alpaca dataset have already been downloaded.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ENV=/nesi/project/uoa04658/rzou895/conda_envs/cross-agent-kv-cache
PYTHON="${ENV}/bin/python"

SAMPLE_COUNT="${SAMPLE_COUNT:-512}"
SEED="${SEED:-1234}"
PERCENTILE="${PERCENTILE:-99.9}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-512}"
OUTPUT="${OUTPUT:-models/qwen05b/calib_general_scales.npz}"

mkdir -p models/qwen05b
mkdir -p results/week02

echo "===== Calibration configuration ====="
echo "Python: ${PYTHON}"
echo "Sample count: ${SAMPLE_COUNT}"
echo "Seed: ${SEED}"
echo "Percentile: ${PERCENTILE}"
echo "Maximum input tokens: ${MAX_INPUT_TOKENS}"
echo "Output: ${OUTPUT}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

echo "===== Checking environment ====="

"${PYTHON}" -c \
'import torch, datasets, numpy, transformers
print("Torch:", torch.__version__)
print("Datasets:", datasets.__version__)
print("NumPy:", numpy.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
print("GPU:", torch.cuda.get_device_name(0))'

echo "===== GPU information ====="
nvidia-smi

echo "===== Collecting calibration statistics ====="

srun "${PYTHON}" -m models.collect_kv_calibration \
    --sample-count "${SAMPLE_COUNT}" \
    --seed "${SEED}" \
    --percentile "${PERCENTILE}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --output "${OUTPUT}"

echo "===== KV calibration completed ====="