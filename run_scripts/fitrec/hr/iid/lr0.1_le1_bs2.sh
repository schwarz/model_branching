#!/bin/sh
python main.py --scale_divisor=220 \
    --window_size=9 \
    --dataset=fitrec --column=heart_rate \
    --model_params="data/fitrec/base_params_hr.pt" \
    --loss_function="mae" \
    --assignment="data/fitrec/assignment_iid.json" \
    --experiment_name="hr_iid_lr0.1_le1_bs2" \
    --learning_rate=3e-5 --local_epochs=1 --mini_batch_size=2 \
    --perfect_node #--disable-cuda # --verbose
