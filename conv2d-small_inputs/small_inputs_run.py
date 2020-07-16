import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa

# from ptflops import get_model_complexity_info

logger = logging.getLogger(__name__)


def run(point):
    start = time.time()
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
            batch_size * image_size * image_size * in_channels,
            dtype=dtype,
            device=device,
        ).view((batch_size, in_channels, image_size, image_size))

        layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                                #padding="same"
        ).to(device, dtype=dtype)

        ave_time = benchmark_forward(layer, inputs)

        # flops, params = get_model_complexity_info(
        #     layer, tuple(inputs.shape[1:]), as_strings=False
        # )
        # print(flops)
        outputs = layer(inputs)
        total_flop = (
            kernel_size
            * kernel_size
            * in_channels
            * out_channels
            * outputs.shape[-1]
            * outputs.shape[-2]
            * batch_size
        )

        print(outputs.shape)

        print("flop = ", total_flop, "ave_time = ", ave_time)
        ave_flops = total_flop / ave_time * batch_size
        runtime = time.time() - start
        print("runtime=", runtime, "ave_flops=", ave_flops)

        return ave_flops
    except Exception as e:
        import traceback

        print("received exception: ", str(e))
        print(traceback.print_exc())
        print("runtime=", time.time() - start)
        # KGF: random addition...
        # logger.exception("exception raised")
        return 0.0


if __name__ == "__main__":
    point = {
        "batch_size": 991,
        "image_size": 64,
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": 4,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
