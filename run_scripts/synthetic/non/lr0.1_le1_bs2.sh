#!/bin/sh
python main.py --scale_divisor=1 \
    --mini_batch_size=2 --window_size=7 \
    --dataset=synthetic --column=y \
    --assignment="data/synthetic/assignment_non.json" \
    --model_params="data/synthetic/base_params.pt" \
    --loss_function="mae" \
    --experiment_name="non_lr0.1_le1_bs2" \
    --learning_rate=1e-5 --local_epochs=1 --perfect_node #--disable-cuda # --verbose
