#!/bin/sh
python main.py --scale_divisor=10600 \
    --window_size=24 \
    --dataset=demand --column=Watts \
    --model_params="data/demand/base_params.pt" \
    --loss_function="mae" \
    --assignment="data/demand/assignment_non.json" \
    --experiment_name="non_lr0.5_le1_bs4" \
    --learning_rate=1.5e-4 --local_epochs=1 --mini_batch_size=4 \
    --perfect_node #--disable-cuda # --verbose
