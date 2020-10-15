#!/bin/sh
python main.py --scale_divisor=5000 \
    --mini_batch_size=2 --window_size=12 \
    --dataset=turnstile --column=entries \
    --assignment="data/turnstile/assignment_non.json" \
    --model_params="data/turnstile/base_params.pt" \
    --loss_function="mae" \
    --learning_rate=1e-5  --local_epochs=2  \
    --experiment_name="non_lr0.1_le2_bs2" --perfect_node # --disable-cuda --verbose