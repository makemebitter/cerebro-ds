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
EXP_NAME="e2e_pytorchddp"
export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
export NCCL_SOCKET_IFNAME='enp94s0f0'
export GLOO_SOCKET_IFNAME='enp94s0f0'
# cp /local/gphost_list /local/all_hosts
# echo "master" >> /local/all_hosts
source runner_helper.sh
WORKING_DIR=$(pwd)
TOTAL_MSTS=$(python2 get_number_msts)
PRINT_START
echo ${SUB_LOG_DIR}/'$WORKER_NAME.log'
for i in $(seq 0 $((TOTAL_MSTS - 1)))
do 
    echo "model-$i, Start time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
    parallel-ssh -i -h /local/gphost_list -t 0  bash $WORKING_DIR/run_pytorchddp_wrapper.sh ${SUB_LOG_DIR} ${WORKING_DIR} ${i} ${EPOCHS} ''"$OPTIONS"'' 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
    echo "model-$i, End time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log 
    # parallel-ssh -h /local/gphost_list -t 0 -P echo "NO" 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
done
PRINT_END



# PRINT_START
# export PYTHONPATH=''
# export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
# 
# python2 -u run_filesystem_cerebro_standalone.py --logs_root $SUB_LOG_DIR --models_root $MODEL_DIR --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
# PRINT_END