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

# imagenet
RESTART_WAIT () {
    gpstop -a -M fast
    echo "Sleep 5 min"
    sleep 300
    gpstart -a
} 

gpstop -a -M fast
gpstart -a
# # # # # # # # # # # # # # # end-to-end # # # # # # # # # # # # # # # # # # # 
# Normal DB
# export MASTER_DATA_DIRECTORY="/mnt/gpdata_master/gpseg-1"
# bash run_da_cerebro_standalone.sh 
# CAUTION: need to restart cerebro workers after
# bash run_filesystem_cerebro_standalone.sh
# for filename in run_mop run_ctq
# for filename in run_imagenet 
# do
#     bash ${filename}.sh
#     RESTART_WAIT
# done

# bash run_udaf_hyperopt.sh
# bash run_ctq_hyperopt.sh
# bash run_da_cerebro_standalone_hyperopt.sh
# bash run_filesystem_cerebro_standalone_hyperopt.sh
# bash run_hyperopt.sh



# # # # # # # # # # # # # # # scalability # # # # # # # # # # # # # # # # # # # 
# Below uses scalability DB
# export MASTER_DATA_DIRECTORY="/mnt/gpdata_scalability/gpseg-1"

# Hetro
# ---------------------------------------------------------------------------
options="--train_name imagenet_train_data_packed_sampled --valid_name imagenet_valid_data_packed"
echo "SIZE: 8"
echo $options
bash run_ctq_drill_down_hetro.sh "" 1 "" ''"$options"''
RESTART_WAIT
# bash run_udaf_drill_down_hetro.sh "" 1 "" ''"$options"''
# RESTART_WAIT


# for size in 4 2
for size in 6 4 2
do  
    echo "SIZE: ${size}"
    options="--train_name imagenet_train_data_packed_${size}_sampled --valid_name imagenet_valid_data_packed_${size}"
    echo $options
    bash run_ctq_drill_down_hetro.sh "" 1 "" ''"$options"''
    RESTART_WAIT
    # bash run_udaf_drill_down_hetro.sh "" 1 "" ''"$options"''
    # RESTART_WAIT

done


# ---------------------------------------------------------------------------




# Scalability cerebro
# ---------------------------------------------------------------------------
# TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
# LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
# mkdir -p $LOG_DIR


# TIMESTAMP_WITH_SIZE=$TIMESTAMP/8
# echo "SIZE: 8"
# options="--drill_down_scalability --size 8 --train_name imagenet_train_data_packed --valid_name imagenet_valid_data_packed"
# echo $options
# bash run_filesystem_cerebro_standalone.sh $TIMESTAMP_WITH_SIZE 1 "" ''"$options"''
# echo "Sleep 5 min"
# sleep 300

# for size in 2
# do  
#     echo "SIZE: ${size}"
#     LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
#     mkdir -p $LOG_DIR
#     TIMESTAMP_WITH_SIZE=$TIMESTAMP/${size}
#     options="--drill_down_scalability --size ${size} --train_name imagenet_train_data_packed_${size} --valid_name imagenet_valid_data_packed_${size}"
#     echo $options
#     bash run_filesystem_cerebro_standalone.sh $TIMESTAMP_WITH_SIZE 1 "" ''"$options"''
#     echo "Sleep 5 min"
#     sleep 300
# done
# ---------------------------------------------------------------------------







