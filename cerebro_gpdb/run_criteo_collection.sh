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

# da

RESTART_WAIT () {
    gpstop -a -M fast
    echo "Sleep 5 min"
    sleep 300
    gpstart -a
} 

gpstop -a -M fast
gpstart -a
# # # # # # # # # # # # # # # end-to-end # # # # # # # # # # # # # # # # # # # 
options="--criteo --train_name criteo_train_data_packed --valid_name criteo_valid_data_packed"
echo $options
# bash run_da_cerebro_standalone.sh "" 5 "" ''"$options"''
# CAUTION: need to restart cerebro workers after
# bash run_filesystem_cerebro_standalone.sh "" 5 "" ''"$options"''

for filename in run_imagenet run_ctq run_mop
do
    bash ${filename}.sh "" 5 "" ''"$options"''
    RESTART_WAIT
done
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # break down # # # # # # # # # # # # # # # # # #
# run udaf and ctq with homo workload
# options="--criteo --criteo_breakdown --train_name criteo_train_data_packed --valid_name criteo_valid_data_packed"
# echo $options
# for filename in run_mop run_ctq
# do
#     bash ${filename}.sh "" 3 "" ''"$options"''
#     RESTART_WAIT
# done
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# bash run_pytorchddp_da.sh "" 5 "" ''"$options"''
# bash run_filesystem_cerebro_spark.sh "" 5 "" ''"$options"''







