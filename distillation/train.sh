deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2.json