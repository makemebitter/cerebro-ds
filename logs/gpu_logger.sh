#!/usr/bin/env bash
if [ "$1" != "" ]; then
    LOG_DIR="$1"
    mkdir -p $LOG_DIR
else
    LOG_DIR="."
fi

LOG_FILENAME="$LOG_DIR/gpu_utilization_$WORKER_NAME.log"

while true;
do  
    echo `date "+%Y-%m-%d %H:%M:%S"` >> $LOG_FILENAME
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader >> $LOG_FILENAME;
    sleep 1;
done