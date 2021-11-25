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
TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
mkdir -p $LOG_DIR
export MASTER_DATA_DIRECTORY="/mnt/gpdata_master/gpseg-1"
gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
gpstart -a -d $MASTER_DATA_DIRECTORY
bash run_mop.sh $TIMESTAMP 1
echo "Sleep 5 min"
sleep 300
bash run_imagenet.sh $TIMESTAMP 1
echo "Sleep 5 min"
sleep 300
bash run_ctq.sh $TIMESTAMP 1
echo "Sleep 5 min"
