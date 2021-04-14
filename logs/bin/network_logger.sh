#!/usr/bin/env bash
if [ "$1" != "" ]; then
    LOG_DIR="$1"
    mkdir -p $LOG_DIR
else
    LOG_DIR="."
fi
CONTROL_IFNAME="$2"
EXPERIMENT_IFNAME="$3"
LOG_FILENAME="$LOG_DIR/network_$WORKER_NAME.log"
echo '' > $LOG_FILENAME
while true;
do
    echo `date "+%Y-%m-%d %H:%M:%S"` >> $LOG_FILENAME
    msg_control=$(sudo sar -n DEV 1 1 | grep $CONTROL_IFNAME | tail -n 1 | gawk '{print "CONTROL_IFACE: "$2", rxpck/s: "$3", txpck/s: "$4", rxkB/s: "$5", txkB/s: "$6", rxcmp/s: "$7", txcmp/s: "$8", rxmcst/s: "$9", %ifutil: "$10}')
    msg_experiment=$(sudo sar -n DEV 1 1 | grep $EXPERIMENT_IFNAME | tail -n 1 | gawk '{print "EXPERIMENT_IFACE: "$2", rxpck/s: "$3", txpck/s: "$4", rxkB/s: "$5", txkB/s: "$6", rxcmp/s: "$7", txcmp/s: "$8", rxmcst/s: "$9", %ifutil: "$10}')
    echo $msg_control >> $LOG_FILENAME;
    echo $msg_experiment >> $LOG_FILENAME;
done




