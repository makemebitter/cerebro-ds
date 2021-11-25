#!/usr/bin/env bash
# Copyright 2020 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
TIMESTAMP=${1:-`date "+%Y_%m_%d_%H_%M_%S"`}
EPOCHS=${2:-10}
SIZE=${3:-8}
OPTIONS=${4:-""}
HOSTS="/local/gphost_list"
master_ip="10.10.1.1"
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
MODEL_DIR="/mnt/nfs/models/$TIMESTAMP"

SUB_LOG_DIR=$LOG_DIR/$EXP_NAME
mkdir -p $SUB_LOG_DIR
mkdir -p $MODEL_DIR

echo $SUB_LOG_DIR
echo $MODEL_DIR
echo "Clearing master sys cache"
free && sync && echo 3 |sudo tee /proc/sys/vm/drop_caches && free
echo "Clearing workers sys cache"
parallel-ssh -i -h $HOSTS -t 0 'free && sync && echo 3 |sudo tee /proc/sys/vm/drop_caches && free'

SHUTDOWN_SPARK (){
    echo "Shutting down master"
    bash /usr/local/spark/sbin/stop-master.sh && sleep 3
    echo "Shutting down workers"
    parallel-ssh -i -h $HOSTS -t 0 "bash /usr/local/spark/sbin/stop-slave.sh"
}
START_SPARK (){
    echo "Starting master"
    bash /usr/local/spark/sbin/start-master.sh && sleep 3
    echo "Starting workers"
    parallel-ssh -i -h $HOSTS -t 0 "bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077" && sleep 6
}
RESTART_SPARK () {
    echo "Restarting master"
    bash /usr/local/spark/sbin/stop-master.sh && sleep 3
    bash /usr/local/spark/sbin/start-master.sh && sleep 6
    echo "Restarting workers"
    parallel-ssh -i -h $HOSTS -t 0 "bash /usr/local/spark/sbin/stop-slave.sh && bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077" && sleep 15
}
RESTART_CEREBRO () {
    echo "Restarting cerebro"
    parallel-ssh -i -h $HOSTS -t 0 "bash /local/cerebro-greenplum/cerebro_gpdb/run_cerebro_worker.sh >/dev/null 2>&1 &" && sleep 10
}

SECONDS=0
PRINT_START () {
   echo "Running $EXP_NAME ..."
   echo "$EXP_NAME, Start time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
} 
PRINT_END () {
   echo "$EXP_NAME, End time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log 
    echo "$EXP_NAME, TOTAL EXECUTION TIME OVER ALL MST $SECONDS"| tee -a $LOG_DIR/global.log
} 


