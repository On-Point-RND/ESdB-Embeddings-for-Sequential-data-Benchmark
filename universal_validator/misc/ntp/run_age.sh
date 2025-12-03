#!/bin/bash

h=64
mx=3000000
ep=350
bs=128
sl=60

python rnn_baseline_age.py \
    -rnn gru \
    --min-frequency 10 \
    --hidden-dim $h \
    --embedding-dim $h \
    --num-layers 2 \
    --dropout 0.5 \
    --weight-decay 1e-4 \
    --batch-size $bs \
    --lr 1e-3 \
    --epochs $ep \
    --max-transactions $mx \
    --max-seq-length $sl \
    --min-seq-length 2 \
    --train-ratio 0.8 \
    --teacher-forcing-ratio 0.9 \
    --data-dir ./age_data/ \
    --patience 20 \
    --cuda-devices 2