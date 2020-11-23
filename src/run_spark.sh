#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
SUB_LOG_DIR=$LOG_DIR/cerebro-spark
mkdir -p $SUB_LOG_DIR
echo "Running cerebro-spark ..."
SECONDS=0
echo "cerebro-spark, Start time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log
export PYTHONPATH=''
/mnt/py3v/bin/python -u run_spark.py --outroot $SUB_LOG_DIR --run --prepared --epoch 10 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
echo "cerebro-spark, End time `date "+%Y-%m-%d %H:%M:%S"`">>$LOG_DIR/global.log 
echo "cerebro-spark, TOTAL EXECUTION TIME OVER ALL MST $SECONDS">>$LOG_DIR/global.log