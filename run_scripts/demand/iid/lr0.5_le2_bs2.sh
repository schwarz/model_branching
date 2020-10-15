#!/bin/sh
python main.py --scale_divisor=10600 \
    --window_size=24 \
    --dataset=demand --column=Watts \
    --model_params="data/demand/base_params.pt" \
    --loss_function="mae" \
    --assignment="data/demand/assignment_iid.json" \
    --experiment_name="iid_lr0.5_le2_bs2" \
    --learning_rate=1.5e-4 --local_epochs=2 --mini_batch_size=2 \
    --perfect_node #--disable-cuda # --verbose
