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
EXP_NAME="e2e_filesystem_cerebro_standalone"
source runner_helper.sh
RESTART_CEREBRO
PRINT_START
export PYTHONPATH=''
export PYTHONPATH="${PYTHONPATH}:/local:/local/cerebro-greenplum"
python3.7 -u run_filesystem_cerebro_standalone.py --logs_root $SUB_LOG_DIR --models_root $MODEL_DIR --run --num_epochs $EPOCHS $OPTIONS 2>&1 | tee -a ${SUB_LOG_DIR}/client.log
PRINT_END