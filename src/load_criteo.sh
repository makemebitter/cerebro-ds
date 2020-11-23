#!/usr/bin/env bash

TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
SUB_LOG_DIR=$LOG_DIR/load-criteo
mkdir -p $SUB_LOG_DIR
echo "Loading criteo ..."
SECONDS=0
echo "Loading criteo, Start time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log
export PYTHONPATH=''
python3.7 -u load_criteo.py --load --pack 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
echo "Loading criteo, End time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log 
echo "Loading criteo, TOTAL EXECUTION TIME OVER ALL MST $SECONDS">>$LOG_DIR/global.log