import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_torch import cuda_vs_knl, use_knl  # noqa

# from ptflops import get_model_complexity_info


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        in_features = point["in_features"]
        out_features = point["out_features"]
        bias = int(point["bias"]) == 1
        print(point)
        import torch

        device, dtype = cuda_vs_knl(point)

        # KGF: attempt to max-out V100 utilization in nvidia-smi for sustained time
        # batch_size *= 100

        inputs = torch.arange(
            batch_size * in_features, dtype=dtype, device=device
        ).view(
            (batch_size, in_features)
        )  # .type(dtype)
        # manually computing flops from the formulas given here:
        # https://machinethink.net/blog/how-fast-is-my-model/
        total_flop = batch_size * (2 * in_features - 1) * out_features
        layer = torch.nn.Linear(in_features, out_features, bias=bias).to(
            device, dtype=dtype
        )  # dtype=float != torch.float
        outputs = layer(inputs)

        runs = 5
        tot_time = 0.0
        tt = time.time()
        for _ in range(runs):
            outputs = layer(inputs)  # noqa F841
            tot_time += time.time() - tt
            tt = time.time()

        ave_time = tot_time / runs

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
        "batch_size": 10,
        "in_features": 512,
        "out_features": 512,
        # KGF: large problem size for 1st evaluation for testing:
        # "batch_size": 128,
        # "in_features": 4096,
        # "out_features": 4096,
        "bias": 1,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
