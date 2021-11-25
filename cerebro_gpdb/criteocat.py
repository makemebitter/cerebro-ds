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
INPUT_SHAPE = (7306, )
NUM_CLASSES = 2
TOTAL = 12993256
param_grid_criteo = {
    "learning_rate": [1e-3, 1e-4],
    "lambda_value": [1e-4, 1e-5],
    "batch_size": [32, 64, 256, 512],
    "model": ["confA"]
}

param_grid_criteo_breakdown = {
    "learning_rate": [1e-3, 1e-4],
    "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
    "batch_size": [256],
    "model": ["confA"]
}

class spark_criteo_cat_nfs:
    valid_list = [
        "/mnt/nfs/hdd/criteo/parquet/valid/valid_{}.parquet".format(i) 
        for i in range(8)
    ]
    train_list = [
        "/mnt/nfs/hdd/criteo/parquet/train/train_{}.parquet".format(i) 
        for i in range(8)
    ]