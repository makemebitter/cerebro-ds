for method in udaf ctq da cerebro_spark cerebro
do
    echo "$method"
    cloc --by-file --list-file=${method}_filelist
done
