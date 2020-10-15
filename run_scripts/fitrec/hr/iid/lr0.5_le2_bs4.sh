#!/bin/sh
python main.py --scale_divisor=220 \
    --window_size=9 \
    --dataset=fitrec --column=heart_rate \
    --model_params="data/fitrec/base_params_hr.pt" \
    --loss_function="mae" \
    --assignment="data/fitrec/assignment_iid.json" \
    --experiment_name="hr_iid_lr0.5_le2_bs4" \
    --learning_rate=1.5e-4 --local_epochs=2 --mini_batch_size=4 \
    --perfect_node #--disable-cuda # --verbose
