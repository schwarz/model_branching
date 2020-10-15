#!/bin/sh
python main.py --scale_divisor=1 \
    --mini_batch_size=4 --window_size=7 \
    --dataset=synthetic --column=y \
    --assignment="data/synthetic/assignment_iid.json" \
    --model_params="data/synthetic/base_params.pt" \
    --loss_function="mae" \
    --experiment_name="iid_lr0.5_le2_bs4" \
    --learning_rate=5e-5 --local_epochs=2 --perfect_node #--disable-cuda # --verbose