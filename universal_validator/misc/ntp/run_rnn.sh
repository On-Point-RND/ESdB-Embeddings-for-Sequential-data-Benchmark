#!/bin/bash

h=128
mx=3000
ep=3000
bs=128
sl=60
minsl=2
pt=9999999
nl=3
mf=10
rnn="lstm"
ds="${1:-age}"  # dataset name, default "age"
dpp="../../../embeddings_${ds}_coles.parquet"  # default path
pp="${2:-$dpp}"  # actual path, use $dpp if $2 is not provided
md="./pt_${ds}"

python rnn_baseline.py \
    --rnn-type $rnn \
    --min-frequency $mf \
    --hidden-dim $h \
    --embedding-dim $h \
    --num-layers $nl \
    --dropout 0.5 \
    --weight-decay 1e-4 \
    --batch-size $bs \
    --lr 1e-3 \
    --epochs $ep \
    --max-clients $mx \
    --max-seq-length $sl \
    --min-seq-length $minsl \
    --train-ratio 0.8 \
    --teacher-forcing-ratio 1.0 \
    --parquet-path $pp \
    --patience $pt \
    --model-dir $md \
    --cuda-devices 3
