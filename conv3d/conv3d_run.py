import time
import psutil

import os

# import sys

# import multiprocessing as mp

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa


def print_mem_cpu():
    start = time.time()
    while True:
        mem = psutil.virtual_memory()
        print(
            "[%010d] pid=%010d total_mem=%010d free_mem=%05.2f cpu_usage=%05.2f"
            % (
                time.time() - start,
                os.getpid(),
                mem.total,
                mem.free / mem.total * 100.0,
                psutil.cpu_percent(),
            )
        )
        time.sleep(1)


def run(point):
    start = time.time()
    # memorymon = mp.Process(target=print_mem_cpu)
    # memorymon.start()
    try:
        batch_size = point["batch_size"]
        image_size = point["image_size"]
        in_channels = point["in_channels"]
        out_channels = point["out_channels"]
        kernel_size = point["kernel_size"]
        print(point)
        import torch

        device, dtype = load_cuda_vs_knl(point)

        inputs = torch.arange(
            batch_size * image_size ** 3 * in_channels, dtype=dtype, device=device
        ).view((batch_size, in_channels, image_size, image_size, image_size))

        layer = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        ).to(device, dtype=dtype)

        # layer.eval()
        ave_time = benchmark_forward(layer, inputs)

        outputs = layer(inputs)
        total_flop = (
            kernel_size ** 3
            * in_channels
            * out_channels
            * outputs.shape[-1]
            * outputs.shape[-2]
            * outputs.shape[-3]
            * batch_size
        )
        print("total_flop = ", total_flop, "ave_time = ", ave_time)

        ave_flops = total_flop / ave_time
        runtime = time.time() - start
        print("runtime=", runtime, "ave_flops=", ave_flops)
        # memorymon.terminate()
        # memorymon.join()
        return ave_flops
    except Exception as e:
        import traceback

        print("received exception: ", str(e), "for point: ", point)
        print(traceback.print_exc())
        print("runtime=", time.time() - start)
        # memorymon.terminate()
        # memorymon.join()
        return 0.0


if __name__ == "__main__":
    point = {
        "batch_size": 32,
        "image_size": 94,
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 9,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
