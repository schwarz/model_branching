#!/bin/sh
python main.py --scale_divisor=5000 \
    --mini_batch_size=2 --window_size=12 \
    --dataset=turnstile --column=entries \
    --assignment="data/turnstile/assignment_iid.json" \
    --model_params="data/turnstile/base_params.pt" \
    --loss_function="mae" \
    --learning_rate=1e-5  --local_epochs=1  \
    --experiment_name="iid_lr0.1_le1_bs2" --perfect_node # --disable-cuda --verbose