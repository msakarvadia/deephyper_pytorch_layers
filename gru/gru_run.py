import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl, use_cuda  # noqa
from utils import get_first_gpu_memory_usage  # noqa
# from ptflops import get_model_complexity_info


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        seq_length = point["seq_length"]
        in_features = point["in_features"]
        hidden_units = point["hidden_units"]
        num_layers = point["num_layers"]
        # out_features = point["out_features"]
        bias = int(point["bias"]) == 1
        print(point)
        import torch

        device, dtype = load_cuda_vs_knl(point)

        init_mem = None
        if use_cuda:
            init_mem = get_first_gpu_memory_usage()

        inputs = torch.arange(
            seq_length * batch_size * in_features, dtype=dtype, device=device,
            requires_grad=True
        ).view(
            (seq_length, batch_size, in_features)
        )

        layer = torch.nn.GRU(in_features, hidden_units, num_layers,
                             # out_features,
                             bias=bias).to(device, dtype=dtype)

        ave_time = benchmark_forward(layer, inputs, init_mem=init_mem)

        # See Dey (2017), GRU has 3*(n^2 + m*n + n) trainable parameters across
        # 6x matrices and 3x bias vectors (size 1xn), where m= input dim, n= hidden dim
        #
        # Or, consider only matrix-vector mults, using 3x combined matrices:
        # [x, h]*A, for A=W_z, W_r, W all (m+n) x n, vector is 1x(m+n)
        # ---> 3*(m+n)*n MACs
        # ---> 3*(2*(m+n) - 1)*n FLOPs
        total_flop = 3 * seq_length * batch_size * (2 * (in_features + hidden_units) - 1)*hidden_units
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
    point = {
        # "batch_size": 10,
        # "seq_length": 10,
        "batch_size": 2048,
        "seq_length": 128,
        # "in_features": 512,
        # "hidden_units": 256,
        "in_features": 1024,
        "hidden_units": 1024,
        "num_layers": 1,
        # "out_features": 512,
        "bias": 1,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
