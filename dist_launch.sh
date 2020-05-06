#!/bin/bash

if [ $# -ne 3 ]; then
    echo "input argument counts is $#, which requires 3, means node_count, hostfile, dtype, to exit"
    exit
fi

node_count=$1
hostfile=$2
dtype=$3

workspace=/mnt/truenas/scratch/xiaotao.chen/Repositories/example_for_hvd_kv

cd ${workspace}

python3 /mnt/truenas/scratch/xiaotao.chen/Repositories/lancher/tools/launch.py \
    -n ${node_count} \
    -s ${node_count} \
    --launcher ssh \
    -H ${hostfile} \
    ./train_dist.sh False ${dtype}
    

