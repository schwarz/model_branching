#!/bin/sh
python main.py --scale_divisor=220 \
    --window_size=9 \
    --dataset=fitrec --column=heart_rate \
    --model_params="data/fitrec/base_params_hr.pt" \
    --loss_function="mae" \
    --assignment="data/fitrec/assignment_non.json" \
    --experiment_name="hr_non_lr0.5_le2_bs2" \
    --learning_rate=1.5e-4 --local_epochs=2 --mini_batch_size=2 \
    --perfect_node #--disable-cuda # --verbose
