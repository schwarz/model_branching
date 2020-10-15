#!/bin/sh
python main.py --scale_divisor=50 \
    --window_size=9 \
    --dataset=fitrec --column=derived_speed \
    --model_params="data/fitrec/base_params_ds.pt" \
    --loss_function="mae" \
    --assignment="data/fitrec/assignment_iid.json" \
    --experiment_name="ds_iid_lr0.5_le1_bs4" \
    --learning_rate=2.5e-4 --local_epochs=1 --mini_batch_size=4 \
    --perfect_node #--disable-cuda # --verbose
