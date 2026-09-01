#!/usr/bin/env bash
set -euo pipefail

n_gpus=1

dataset="${1:-alpha}"
method="${2:-jepa}"
array_range="${3:-0-0}"
n_days="${4:-6}"
login="${5:-e.surkov}"
validator="${6:-universal_validator/configs/validator/logreg.yaml}"

partition="${PARTITION:-ais-gpu}"
image="${IMAGE:-image_trans.sif}"
project_dir="${PROJECT_DIR:-/home/ESdB-Embeddings-for-Sequential-data-Benchmark}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${OUTPUT_DIR:-${root}/zhores/outputs/optuna/${dataset}/${method}}"

cd "${root}"

if [[ ! -f "configs/datasets/${dataset}.yaml" ]]; then
    echo "Unknown dataset config: ${dataset}" >&2
    exit 1
fi
if [[ ! -f "configs/methods/${method}.yaml" ]]; then
    echo "Unknown method config: ${method}" >&2
    exit 1
fi
if [[ ! -f "${validator}" ]]; then
    echo "Validator config not found: ${validator}" >&2
    exit 1
fi

mkdir -p "${output_dir}"

submission="$({ sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=optuna_${dataset}_${method}
#SBATCH --partition=${partition}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${login}@skoltech.ru
#SBATCH --array=${array_range}
#SBATCH --output=${output_dir}/%A_%a.out
#SBATCH --error=${output_dir}/%A_%a.err
#SBATCH --time=${n_days}-00
#SBATCH --mem=$((n_gpus * 100))G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$((8 * n_gpus))
#SBATCH --gpus=${n_gpus}

set -euo pipefail

echo "host: \$(hostname)"
echo "job: \${SLURM_JOB_ID}, array task: \${SLURM_ARRAY_TASK_ID}"
nvidia-smi

srun singularity exec \
    --bind "/gpfs/gpfs0/${login}:/home" \
    --nv \
    "${image}" \
    bash -lc '
set -euo pipefail
cd "${project_dir}"
export PYTHONUNBUFFERED=1
echo "command: python main.py -d ${dataset} -m ${method} -e optuna -dv ${validator} -g cuda:0"
python main.py -d "${dataset}" -m "${method}" -e optuna -dv "${validator}" -g cuda:0
'
EOT
} 2>&1)"

echo "${submission}"
job_id="${submission##* }"
echo "Slurm queue: squeue -j ${job_id}"
echo "Slurm stdout: ${output_dir}/${job_id}_<array_task>.out"
echo "Slurm stderr: ${output_dir}/${job_id}_<array_task>.err"
echo "Experiment logs: ${project_dir}/log/${dataset}/${method}/optuna"
