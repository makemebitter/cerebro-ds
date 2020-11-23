#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
EXP_NAME="cerebro-udaf_drill_down_hetro"
SUB_LOG_DIR=$LOG_DIR/$EXP_NAME
mkdir -p $SUB_LOG_DIR
echo "Running $EXP_NAME ..."
SECONDS=0
echo "$EXP_NAME, Start time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log
export PYTHONPATH=''
python3.7 -u run_imagenet.py --load --drill_down_hetro 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
python3.7 -u run_mop.py --load --run --num_epochs 1 --drill_down_hetro 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
echo "$EXP_NAME, End time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log 
echo "$EXP_NAME, TOTAL EXECUTION TIME OVER ALL MST $SECONDS">>$LOG_DIR/global.log