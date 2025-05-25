deepspeed --include localhost:1,2,3,4 train.py \
    --deepspeed configs/ds_config_zero2.json