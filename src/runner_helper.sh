#!/usr/bin/env bash
TIMESTAMP=${1:-`date "+%Y_%m_%d_%H_%M_%S"`}
EPOCHS=${2:-10}
SIZE=${3:-8}
OPTIONS=${4:-""}

LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
MODEL_DIR="/mnt/nfs/models/$TIMESTAMP"

SUB_LOG_DIR=$LOG_DIR/$EXP_NAME
mkdir -p $SUB_LOG_DIR
mkdir -p $MODEL_DIR

echo $SUB_LOG_DIR
echo $MODEL_DIR

SECONDS=0
PRINT_START () {
   echo "Running $EXP_NAME ..."
   echo "$EXP_NAME, Start time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
} 
PRINT_END () {
   echo "$EXP_NAME, End time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log 
    echo "$EXP_NAME, TOTAL EXECUTION TIME OVER ALL MST $SECONDS"| tee -a $LOG_DIR/global.log
} 