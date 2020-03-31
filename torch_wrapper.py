import os
import time
import torch
from utils import get_first_gpu_memory_usage

num_runs = 5  # number of independent trials (or forward passes)
use_cuda = torch.cuda.is_available()
use_knl = False
# TODO(KGF): more robust way to detect presence of an Intel KNL device?
# this might be unset for the default affinity? Check hostname for Theta instead?
#
# export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
#
# print(f'os.environ["KMP_AFFINITY"] = {os.environ.get("KMP_AFFINITY")}')
#
# if os.environ.get("KMP_AFFINITY") is not None:
if os.environ.get("CRAY_CPU_TARGET") == "mic-knl":
    use_knl = True


def load_cuda_vs_knl(point):
    print("torch version: ", torch.__version__, " torch file: ", torch.__file__)

    print("PyTorch: CUDA available? {}".format(use_cuda))
    if use_cuda:
        assert not use_knl
        print("torch.cuda.current_device() = {}".format(torch.cuda.current_device()))
        print(
            "CUDA_DEVICE_ORDER = {}".format(
                os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
            )
        )
        print("CUDA_VISIBLE_DEVICES = {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # print("torch.cuda.device(0) = {}".format(torch.cuda.device(0)))
        # print("torch.cuda.device_count() = {}".format(torch.cuda.device_count()))
        # print("torch.cuda.get_device_name(0) = {}".format(
        #     torch.cuda.get_device_name(0)))
        device = torch.device("cuda")
        # https://pytorch.org/docs/stable/tensors.html
        dtype = torch.float  # equivalent to torch.float32
        # dtype = torch.float16

        # dtype = torch.cuda.FloatTensor
        torch.backends.cudnn.benchmark = True
    else:
        assert use_knl
        # TODO(KGF): assuming KNL if not CUDA GPU; add KNL vs. "regular CPU" switch
        omp_num_threads = point["omp_num_threads"]
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_num_threads)
        # New warning on Theta when using KMP_HW_SUBSET:
        #
        # OMP: Warning #244: KMP_HW_SUBSET: invalid value "1s,64c,2t", valid format is
        # "N<item>[@N][,...][,Nt] (<item> can be S, N, L2, C, T  for Socket, NUMA Node,
        # L2 Cache, Core, Thread)".
        #
        # os.environ["KMP_HW_SUBSET"] = "1s,%sc,2t" % str(omp_num_threads)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["MKLDNN_VERBOSE"] = str(1)
        # KGF: strange bug discovered on Theta aroudn 2020-02-23: setting MKL_VERBOSE=1
        # BEFORE import torch causes seg-fault when Python interpreter exits
        os.environ["MKL_VERBOSE"] = str(1)
        device = torch.device("cpu")
        dtype = torch.float32
        # dtype = torch.FloatTensor
    return device, dtype


def benchmark_forward(layer, inputs, init_mem=None):
    tt = time.time()
    outputs = layer(inputs)
    print("Time for initial PyTorch layer evaluation = {} s".format(time.time() - tt))
    tot_time = 0.0
    tot_mem = 0.0
    for i in range(num_runs):
        if use_cuda:
            torch.cuda.synchronize()
        tt = time.time()
        # outputs = layer(inputs).detach()  # noqa F841
        outputs = layer(inputs)  # noqa F841
        if use_cuda:
            torch.cuda.synchronize()
        t_run = time.time() - tt
        tot_time += t_run
        print(f"Run {i}: {t_run} s")
        if init_mem is not None:
            tot_mem += get_first_gpu_memory_usage() - init_mem
            print(f"    {get_first_gpu_memory_usage() - init_mem} Mb")
        # del outputs

    ave_time = tot_time / num_runs
    if init_mem is not None:
        ave_mem = tot_mem / num_runs
        print(f"Initial GPU memory usage: {init_mem:.2f}")
        print(f"Average GPU memory used: {ave_mem:.2f}")
    # TODO: report std-dev and mean, not just mean

    return ave_time
