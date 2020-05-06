import mxnet as mx
import numpy as np
import horovod.mxnet as hvd
import os
from mxnet import optimizer as opt
import pickle as pkl
import time
import argparse

def test_hvd_kv(rank, num_workers, kv, dtype="float32"):
    assert dtype in ("float32", "float16")
    np.random.seed(5+rank)
    gpu_num = 8
    tensor_count = 10
    repeat_count = 10
    # shape = (1,1,2,3)
    shape = (128,1024,3,3)
    params_names = ["w_{}".format(i) for i in range(tensor_count)]
    params_shapes = [shape[:3] + (3+i,) for i in range(tensor_count)]
    params_array = [[mx.nd.zeros(shape=params_shapes[t_id], ctx=mx.gpu(i)) for i in range(8)] for t_id in range(tensor_count)]
    for idx in range(tensor_count):
        kv.init(params_names[idx], params_array[idx][0])
    for cnt in range(repeat_count):
        print("{}/{} update {}...".format(rank, num_workers, cnt))
        for idx in range(tensor_count):
            # grad_list = [mx.nd.ones(shape=shape, ctx=mx.gpu(i), dtype=dtype) * (rank * gpu_num + i) for i in range(gpu_num)]
            grad_list = [mx.nd.array(np.random.uniform(size=params_shapes[idx]) * 10, ctx=mx.gpu(i), dtype=dtype) * (rank * gpu_num + i) for i in range(gpu_num)]
            arg_list = params_array[idx]
            name = params_names[idx]

            kv.push(name, grad_list)
            kv.pull(name, arg_list)
            mx.nd.waitall()
    
    if rank == 0:
        params_dict = {}
        for idx in range(tensor_count):
            params_dict[params_names[idx]] = params_array[idx][0].asnumpy()
        pkl_name = "{}_{}.pkl".format('hvd' if use_horovod else "kv_dist", dtype)
        with open(pkl_name, 'wb') as f:
            pkl.dump(params_dict, f)
    
    time.sleep(2)

def test_allreduce(use_horovod, dtype):
    if use_horovod is False:
        kvstore_type = "dist_sync_device" if os.environ.get("DMLC_ROLE") == "worker" else kvstore_type
        kv = mx.kvstore.create(kvstore_type)
        rank = kv.rank
        num_workers = kv.num_workers
    else:
        kvstore_type = "device"
        kv = mx.kvstore.create(kvstore_type)
        hvd.init()
        rank = hvd.rank()
        num_workers = hvd.size()
    print('use horovod: {}, rank {}/{}, kv type: {}, usetree: {}'.format(
           use_horovod, rank, num_workers, kvstore_type, 
           os.environ.get("MXNET_KVSTORE_USETREE")))

    rescale_grad = 1.0 / (8 * num_workers)
    if use_horovod:
        rescale_grad = rescale_grad * num_workers

    optimizer_params = dict(
        momentum=0, # pOpt.optimizer.momentum,
        wd=0, # pOpt.optimizer.wd,
        learning_rate=0.1,
        rescale_grad=rescale_grad,
    )
    optimizer = mx.optimizer.create("sgd", **optimizer_params)
    if use_horovod:
        # Horovod: wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(optimizer)

    print("opt rescale:{}".format(optimizer.rescale_grad))
    kv.set_optimizer(optimizer)

    test_hvd_kv(rank, num_workers, kv, dtype)

def check_result(dtype):
    hvd_file = "hvd_{}.pkl".format(dtype)
    kv_file = "kv_dist_{}.pkl".format(dtype)
    with open(hvd_file, 'rb') as f:
        hvd_params = pkl.load(f)
    with open(kv_file, 'rb') as f:
        kv_params = pkl.load(f)
    
    for k,v in hvd_params.items():
        assert k in kv_params.keys()
        print("check {} : {}, dtype:{}".format(k, v.shape, dtype))
        # don't set decimal=6, this will cause mismatch, this mismatch may be caused by calculation precision difference bewteen kvstore and hvd or optimizer.
        # this mismatch have no relative with allreduce operation in hvd or kv.
        np.testing.assert_almost_equal(kv_params[k], hvd_params[k], decimal=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='example for hvd kv')
    parser.add_argument('--use_horovod', help='use horovod or not', type=str)
    parser.add_argument("--dtype", help="data type float32 or float16", type=str)
    args = parser.parse_args()
    use_horovod = eval(args.use_horovod)
    dtype = args.dtype # "float32", "float16"
    print("[example for hvd kv] use horovod:{}, dtype:{}".format(use_horovod, dtype))
    test_allreduce(use_horovod, dtype)
    # check_result(dtype)
    