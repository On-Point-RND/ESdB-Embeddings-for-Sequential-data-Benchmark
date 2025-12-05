#!/bin/bash

h=64
mx=1000
ep=500
bs=128
sl=60
pt=200
rnn="lstm"
pp="../../../embeddings_age_coles.parquet"
pp="../../../embeddings_age_x5.parquet"


python rnn_baseline.py \
    --rnn-type $rnn \
    --min-frequency 100 \
    --hidden-dim $h \
    --embedding-dim $h \
    --num-layers 2 \
    --dropout 0.5 \
    --weight-decay 1e-4 \
    --batch-size $bs \
    --lr 1e-3 \
    --epochs $ep \
    --max-clients $mx \
    --max-seq-length $sl \
    --min-seq-length 2 \
    --train-ratio 0.8 \
    --teacher-forcing-ratio 0.9 \
    --parquet-path $pp \
    --patience $pt \
    --cuda-devices 2