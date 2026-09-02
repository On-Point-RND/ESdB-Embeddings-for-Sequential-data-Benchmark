#!/usr/bin/env bash
#SBATCH --job-name=esdb-ablation
#SBATCH --partition=ais-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --gpus=8
#SBATCH --time=6-00:00:00
#SBATCH --output=/trinity/home/e.surkov/jupyter-%j.log
#SBATCH --error=/trinity/home/e.surkov/jupyter-%j.log
#SBATCH --open-mode=append

set -euo pipefail

report_exit() {
    status=$?
    echo "Job finished at $(date --iso-8601=seconds) with exit code ${status}"
}
trap report_exit EXIT

HOST_DEV_DIR="/trinity/home/e.surkov/dev"
PROJECT_NAME="ESdB-Embeddings-for-Sequential-data-Benchmark"
PROJECT_DIR="${HOST_DEV_DIR}/${PROJECT_NAME}"
CONTAINER_PROJECT_DIR="/home/${PROJECT_NAME}"
IMAGE="${HOST_DEV_DIR}/image_trans.sif"
PORT="${PORT:-7777}"
TOKEN="${JUPYTER_TOKEN:-$(printf '%s' "${SLURM_JOB_ID}-$(date +%s%N)" | sha256sum | cut -c1-32)}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi

if [[ ! -f "${IMAGE}" ]]; then
    echo "Container image not found: ${IMAGE}" >&2
    exit 1
fi

NODE="${SLURMD_NODENAME:-$(hostname)}"

echo "Job ID:               ${SLURM_JOB_ID}"
echo "Node:                 ${NODE}"
echo "Slurm GPU IDs:        ${SLURM_JOB_GPUS:-unknown}"
echo "Job log:               /trinity/home/e.surkov/jupyter-${SLURM_JOB_ID}.log"
echo "SSH tunnel:            ssh -N -L ${PORT}:${NODE}:${PORT} e.surkov@<cluster-login-host>"
echo "Open locally:          http://localhost:${PORT}/lab?token=${TOKEN}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_HF_HOME=/home/.cache/huggingface

srun \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    singularity exec \
        --nv \
        --bind "${HOST_DEV_DIR}:/home" \
        "${IMAGE}" \
        env HOME=/home jupyter-lab \
            --port "${PORT}" \
            --ip 0.0.0.0 \
            --ServerApp.root_dir="${CONTAINER_PROJECT_DIR}" \
            --IdentityProvider.token="${TOKEN}" \
            --ServerApp.password=''
