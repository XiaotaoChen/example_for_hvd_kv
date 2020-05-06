#!/bin/bash
eval $(cd && .tspkg/bin/tsp --env)

if [ $# -ne 2 ]; then
    echo "input argument counts is $#, which requires 2, means use_horovod, dtype, to exit"
    exit
fi

use_horovod=$1
dtype=$2

workspace=/mnt/truenas/scratch/xiaotao.chen/Repositories/example_for_hvd_kv

cd ${workspace}

export PYTHONPATH=$PYTHONPATH:/mnt/truenas/scratch/xiaotao.chen/Repositories/lancher/mxnet_xyxy/python
export MXNET_KVSTORE_USETREE=1
# if your use hvd for distributed training, ensure setting this. or the performance will be bad.
# export MXNET_UPDATE_ON_KVSTORE=1

# export MXNET_EXEC_BULK_EXEC_INFERENCE=0
# export MXNET_EXEC_BULK_EXEC_TRAIN=0
# export MXNET_PROFILER_AUTOSTART=1

python3  example_for_hvd_kv.py --use_horovod ${use_horovod} --dtype ${dtype}