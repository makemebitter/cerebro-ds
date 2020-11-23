#!/usr/bin/env bash
export EXP_NAME=cerebro
source runner_helper.sh

PRINT_START
PYTHONPATH=''
echo $OPTIONS
python3.7 -u run_imagenet.py --load $OPTIONS 2>&1 | tee ${SUB_LOG_DIR}/client.log
python3.7 -u run_mop.py --load --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee ${SUB_LOG_DIR}/client.log
PRINT_END