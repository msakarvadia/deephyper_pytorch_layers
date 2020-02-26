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
        height = point["height"]
        width = point["width"]
        in_channels = point["in_channels"]
        out_channels = point["out_channels"]
        kernel_size = point["kernel_size"]
        print(point)
        import torch

        device, dtype = cuda_vs_knl(point)

        inputs = torch.arange(
            batch_size * height * width * in_channels, dtype=dtype, device=device
        ).view((batch_size, in_channels, height, width))

        layer = torch.nn.Conv2d(
            in_channels, out_channels, (kernel_size, kernel_size), stride=1, padding=1
        ).to(device, dtype=dtype)
        # flops, params = get_model_complexity_info(layer,
        #         tuple(inputs.shape[1:]),as_strings=False)
        # print('ptflops=',flops*batch_size)
        outputs = layer(inputs)
        # print('shapes: ',inputs.shape,outputs.shape)
        total_flop = (
            kernel_size
            * kernel_size
            * in_channels
            * out_channels
            * outputs.shape[-1]
            * outputs.shape[-2]
            * batch_size
        )
        # print('total_flop=',total_flop)

        runs = 5
        tot_time = 0.0
        tt = time.time()
        for i in range(runs):
            outputs = layer(inputs)
            tot_time += time.time() - tt
            tt = time.time()

        ave_time = tot_time / runs

        print("total_flop = ", total_flop, "ave_time = ", ave_time)

        ave_flops = total_flop / ave_time
        runtime = time.time() - start
        print("runtime=", runtime, "ave_flops=", ave_flops)

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
        "height": 32,
        "width": 32,
        "in_channels": 3,
        "out_channels": 6,
        "kernel_size": 5,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
