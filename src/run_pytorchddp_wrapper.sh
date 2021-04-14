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
SUB_LOG_DIR=${1}
WORKING_DIR=${2}
i=${3}
EPOCHS=${4}
OPTIONS=${5:-""}
echo "In wrapper"
export NCCL_SOCKET_IFNAME='enp94s0f0'
export GLOO_SOCKET_IFNAME='enp94s0f0'
# kill any leftover
while $(killall -9 python2 2>/dev/null); do 
    sleep 1
done

sleep 3
cd $WORKING_DIR
export PYTHONPATH=''
export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
echo "Clearing sys cache"
free && sync && echo 3 |sudo tee /proc/sys/vm/drop_caches && free
python2 -u run_pytorchddp.py --logs_root ${SUB_LOG_DIR} --num_epochs ${EPOCHS} --run_single --single_mst_index ${i} ${OPTIONS} 2>&1 | tee -a ${SUB_LOG_DIR}/${WORKER_NAME}.log