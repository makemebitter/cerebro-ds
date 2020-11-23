#!/usr/bin/env bash
export EXP_NAME="cerebro-da"
source runner_helper.sh

PRINT_START
export PYTHONPATH=''
export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
python2 -u run_da_cerebro_standalone.py --logs_root $SUB_LOG_DIR --models_root $MODEL_DIR --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
PRINT_END