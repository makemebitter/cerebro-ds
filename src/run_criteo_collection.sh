#!/usr/bin/env bash

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
bash run_da_cerebro_standalone.sh "" 5 "" ''"$options"''
bash run_filesystem_cerebro_standalone.sh "" 5 "" ''"$options"''
for filename in run_imagenet run_ctq run_mop
do
    bash ${filename}.sh "" 5 "" ''"$options"''
    RESTART_WAIT
done
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # break down # # # # # # # # # # # # # # # # # #
# run udaf and ctq with homo workload
options="--criteo --criteo_breakdown --train_name criteo_train_data_packed --valid_name criteo_valid_data_packed"
echo $options
for filename in run_mop run_ctq
do
    bash ${filename}.sh "" 3 "" ''"$options"''
    RESTART_WAIT
done
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #








