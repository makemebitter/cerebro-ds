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
export EXP_NAME=cerebro
source runner_helper.sh

PRINT_START
export PYTHONPATH=''
export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
echo $OPTIONS
python3.7 -u run_imagenet.py --load $OPTIONS 2>&1 | tee ${SUB_LOG_DIR}/client.log
python3.7 -u run_mop.py --load --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee ${SUB_LOG_DIR}/client.log
PRINT_END