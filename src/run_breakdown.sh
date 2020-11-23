TIMESTAMP=`date "+%Y_%m_%d_%H_%M_%S"`
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
mkdir -p $LOG_DIR
export MASTER_DATA_DIRECTORY="/mnt/gpdata_master/gpseg-1"
gpstop -d $MASTER_DATA_DIRECTORY -a -M fast
gpstart -a -d $MASTER_DATA_DIRECTORY
# bash run_mop.sh $TIMESTAMP 1
# echo "Sleep 5 min"
# sleep 300
# bash run_imagenet.sh $TIMESTAMP 1
# echo "Sleep 5 min"
# sleep 300
bash run_ctq.sh $TIMESTAMP 1
echo "Sleep 5 min"
