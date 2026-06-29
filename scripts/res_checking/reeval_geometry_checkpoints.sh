#!/usr/bin/env bash
set -euo pipefail

GEOMETRY_KIND="${GEOMETRY_KIND:-jepa}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-rsample ${GEOMETRY_KIND}_geometry_export}"
VALIDATOR="${VALIDATOR:-universal_validator/configs/validator/${GEOMETRY_KIND}_geometry_metrics.yaml}"

export EXTRA_CONFIGS VALIDATOR
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
exec "${SCRIPT_DIR}/reeval_checkpoints_parallel.sh"
