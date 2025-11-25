h=64
mx=500000
ep=100
python rnn_baseline_age.py \
    --split-method strict_temporal \
    --min-frequency 10 \
    --hidden-dim $h \
    --embedding-dim $h \
    --num-layers 2 \
    --dropout 0.5 \
    --weight-decay 1e-4 \
    --batch-size 64 \
    --lr 1e-3 \
    --epochs $ep \
    --max-transactions $mx \
    --sequence-length 5 \
    --train-ratio 0.8 \
    --teacher-forcing-ratio 0.8 \
    --data-dir ./age_data/ \
    --cuda-devices 2