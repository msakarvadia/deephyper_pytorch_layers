import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl, use_cuda  # noqa
from utils import get_first_gpu_memory_usage
# from ptflops import get_model_complexity_info
# https://github.com/sovrasov/flops-counter.pytorch
#
# from thop import profile
# https://github.com/Lyken17/pytorch-OpCounter


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        in_features = point["in_features"]
        out_features = point["out_features"]
        bias = int(point["bias"]) == 1
        print(point)
        import torch

        device, dtype = load_cuda_vs_knl(point)

        # KGF: check GPU memory usage baseline after loading PyTorch, but before any
        # inputs, model, etc. are defined
        init_mem = None
        if use_cuda:
            init_mem = get_first_gpu_memory_usage()

        # KGF: attempt to max-out V100 utilization in nvidia-smi for a sustained time:
        # batch_size *= 100

        inputs = torch.arange(
            batch_size * in_features, dtype=dtype, device=device, requires_grad=True
        ).view(
            (batch_size, in_features)
        )  # .type(dtype)
        # manually computing flops from the formulas given here:
        # https://machinethink.net/blog/how-fast-is-my-model/
        #
        # FC layer: x*W, x is 1xI vector, W is IxJ matrix
        # ----> I*J MACs
        # dot product (specifically) of n MACs = n multiplications, n-1 adds
        # FC layer computes J dot products:
        # ----> (2I-1)*J FLOPs
        total_flop = batch_size * (2 * in_features - 1) * out_features
        layer = torch.nn.Linear(in_features, out_features, bias=bias).to(
            device, dtype=dtype
        )  # dtype=float != torch.float

        ave_time = benchmark_forward(layer, inputs, init_mem=init_mem)
        print("flop = ", total_flop, "ave_time = ", ave_time)
        ave_flops = total_flop / ave_time

        print("runtime=", time.time() - start, "ave_flops=", ave_flops)

        return ave_flops
    except Exception as e:
        import traceback

        print("received exception: ", str(e))
        print(traceback.print_exc())
        print("runtime=", time.time() - start)
        return 0.0


if __name__ == "__main__":
    # batch_size,bias,in_features,out_features,objective,elapsed_sec
    # 6871,0,8153,7533,1047811754548361.0,1037.2227339744568

    point = {
        "batch_size": 6871,
        "in_features": 8153,
        "out_features": 7533,
        "bias": 0,
        #
        # "batch_size": 10,
        # "in_features": 512,
        # "out_features": 512,
        #
        # KGF: large problem size for 1st evaluation for testing:
        # "batch_size": 128,
        # "in_features": 4096,
        # "out_features": 4096,
        # "bias": 1,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
