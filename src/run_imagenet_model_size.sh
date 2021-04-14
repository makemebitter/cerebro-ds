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
trap "exit" INT TERM ERR
trap "kill 0" EXIT

TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
UDAF_LOG_DIR="$LOG_DIR/udaf"
CTQ_LOG_DIR="$LOG_DIR/ctq"
MODEL_DIR="/mnt/nfs/models/$TIMESTAMP"
mkdir -p $UDAF_LOG_DIR
mkdir -p $CTQ_LOG_DIR
mkdir -p $MODEL_DIR
EXP_NAME="drill_down_udaf_ctq_model_size"
NUM_EPOCHS=3

echo "Restarting gpdb ..."
export PYTHONPATH="/usr/local/gpdb/lib/python"
gpstop -a -M fast
gpstart -a
echo "Completed restarting gpdb"

for mode in "udaf" "ctq" 
do
    for model_size in "s" "m" "l" "x";
    do  
        export PYTHONPATH=''
        SUB_EXP_NAME="$EXP_NAME-$mode-$model_size"
        echo "Running $SUB_EXP_NAME ..."
        SECONDS=0
        echo "$SUB_EXP_NAME, Start time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
        
        if [ $mode == "udaf" ];then
            curr_exp_log_dir=$UDAF_LOG_DIR/"model_size_${model_size}"
            mkdir -p $curr_exp_log_dir
            echo "Running udaf ..."
            python3.7 -u run_imagenet.py \
            --load \
            --drill_down_model_size \
            --drill_down_model_size_identifier $model_size \
            2>&1 | tee $curr_exp_log_dir/client.log
            sleep 20
            python3.7 -u run_mop.py \
            --load \
            --run \
            --num_epochs $NUM_EPOCHS \
            --drill_down_model_size \
            --drill_down_model_size_identifier $model_size \
            2>&1 | tee $curr_exp_log_dir/client.log
        else
            curr_exp_log_dir=$CTQ_LOG_DIR/"model_size_${model_size}"
            mkdir -p $curr_exp_log_dir
            echo "Running ctq ..."
            python3.7 -u ctq.py \
            --logs_root $curr_exp_log_dir \
            --models_root $MODEL_DIR \
            --run \
            --num_epochs $NUM_EPOCHS \
            --drill_down_model_size \
            --drill_down_model_size_identifier $model_size \
            2>&1 | tee -a $curr_exp_log_dir/client.log
        fi
        echo "$SUB_EXP_NAME, End time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log 
        echo "$SUB_EXP_NAME, TOTAL EXECUTION TIME OVER ALL MST $SECONDS"| tee -a $LOG_DIR/global.log
        echo "Restarting gpdb ..."
        export PYTHONPATH="/usr/local/gpdb/lib/python"
        gpstop -a -M fast
        gpstart -a
        echo "Completed restarting gpdb"
        echo "Wait 5 min"
        sleep 300
    done
done

