#!/bin/sh
set -e
set -u

d="${1:-twitter}"
g="${2:-3}"
m="${3:-ntp_gpt}"
s="${4:-classification}"
dv="universal_validator/config.yaml"

export CUDA_VISIBLE_DEVICES="$g"
export TASK_NAME="$s"

python main.py -d "full/${d}" -m "$m" -e train_best -s "$s" -dv "$dv"


