#!/usr/bin/env bash
export EXP_NAME="cerebro-ctq"
source runner_helper.sh


PRINT_START
export PYTHONPATH=''
python3.7 -u ctq.py --logs_root $SUB_LOG_DIR --models_root $MODEL_DIR --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
PRINT_END