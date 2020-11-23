#!/usr/bin/env bash
initialize=''
load=''
run=''
print_usage() {
  printf "Usage: ..."
}

while getopts 'ilr' flag; do
  case "${flag}" in
    i) initialize='true' ;;
    l) load='true' ;;
    r) run='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

export PYTHONPATH='/usr/local/gpdb/lib/python'
SHUT_DOWN_ALL () {
    echo "Shutting down 8-node ..."
    gpstop -d "/mnt/gpdata_master/gpseg-1" -a -M fast
    for SHUT_DOWN_ALL_size in 1 2 4
    do
        echo "Shutting down $SHUT_DOWN_ALL_size-node ..."
        gpstop -d "/mnt/gpdata_${SHUT_DOWN_ALL_size}_master/gpseg-1" -a -M fast
    done
} 

if [[ $initialize == 'true' ]] 
then
    for size in 4 2
    do
        echo "Initializing cluster $size"
        gpinitsystem -a -c ../gp_configs/gpinitsystem_config_$size -h ../gp_configs/gphost_list_$size
        export MASTER_DATA_DIRECTORY="/mnt/gpdata_${size}_master/gpseg-1"
        echo "Launching cluster $size"
        gpstart -a -d $MASTER_DATA_DIRECTORY
        echo "Changing settings cluster $size"
        gpconfig -c gp_vmem_protect_limit -v 153600
        gpconfig -c max_statement_mem -v 153600MB
        gpconfig -c statement_mem -v 15360MB
        echo "Shutting down cluster $size"
        gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
        echo "Launching cluster $size"
        gpstart -a -d $MASTER_DATA_DIRECTORY
        echo "Installing madlib cluster $size"
        /local/madlib/build/src/bin/madpack -p greenplum -c gpadmin@master:5432/cerebro install
        bash /local/madlib/tool/cluster_install.sh
        echo "Shutting down cluster $size"
        gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
    done
fi 

for size in 4 2 1
do
    if [[ $load == 'true' ]]
    then
        echo $size
        SHUT_DOWN_ALL
        echo "Loading imagenet"
        export MASTER_DATA_DIRECTORY="/mnt/gpdata_${size}_master/gpseg-1"
        echo "Launching cluster $size"
        gpstart -a -d $MASTER_DATA_DIRECTORY
        echo "Loading imagenet"
        bash load_imagenet.sh
        echo "Clean up tmp files"
        sudo rm -rvf /mnt/madlib_*
        echo "Shutting down cluster $size"
        gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
    fi
done

# da tests have to be done separately with the rest, as cerebro worker does not release GPU memory
# if [[ $run == 'true' ]]
# then
#     TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
#     LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
#     mkdir -p $LOG_DIR
#     for size in 2
#     do
#         SHUT_DOWN_ALL
#         export MASTER_DATA_DIRECTORY="/mnt/gpdata_${size}_master/gpseg-1"
#         echo "Launching cluster $size"
#         gpstart -a -d $MASTER_DATA_DIRECTORY
#         TIMESTAMP_WITH_SIZE="$TIMESTAMP/$size"
#         bash run_da_cerebro_standalone.sh $TIMESTAMP_WITH_SIZE 1 $size "--drill_down_scalability"
#         # echo "Sleep 5 min"
#         # sleep 300
#         # bash run_mop.sh $TIMESTAMP_WITH_SIZE 1 $size "--drill_down_scalability"
#         # echo "Sleep 5 min"
#         # sleep 300
#         # bash run_ctq.sh $TIMESTAMP_WITH_SIZE 1 $size "--drill_down_scalability"
#         # echo "Sleep 5 min"
#         # sleep 300
#         # bash run_imagenet.sh $TIMESTAMP_WITH_SIZE 1 $size "--drill_down_scalability"
#         # echo "Sleep 5 min"
#         # sleep 300
#         echo "Shutting down cluster $size"
#         gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
#     done
# fi

# size one and eight are run differently
if [[ $run == 'true' ]]
then
    TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
    LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
    mkdir -p $LOG_DIR
    for size in 1
    do  
        
        SHUT_DOWN_ALL
        export MASTER_DATA_DIRECTORY="/mnt/gpdata_master/gpseg-1"
        echo "Launching cluster $size"
        gpstart -a -d $MASTER_DATA_DIRECTORY
        TIMESTAMP_WITH_SIZE=$TIMESTAMP/$size
        options='--drill_down_scalability --size '"$size"''
        echo $options
        bash run_da_cerebro_standalone.sh $TIMESTAMP_WITH_SIZE 1 $size ''"$options"''
        echo "Sleep 5 min"
        sleep 300
        # bash run_mop.sh $TIMESTAMP_WITH_SIZE 1 $size ''"$options"''
        # echo "Sleep 5 min"
        # sleep 300
        # bash run_ctq.sh $TIMESTAMP_WITH_SIZE 1 $size ''"$options"''
        # echo "Sleep 5 min"
        # sleep 300
        # bash run_imagenet.sh $TIMESTAMP_WITH_SIZE 1 $size ''"$options"''
        # echo "Sleep 5 min"
        # sleep 300
        echo "Shutting down cluster $size"
        gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
    done

fi

