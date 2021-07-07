import time
import logging
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa

# from ptflops import get_model_complexity_info

logger = logging.getLogger(__name__)


def run(point):
    start = time.time()
    try:
        out_channels = point["out_channels"]
        batch_size_1 = point["batch_size"]
        image_size_1 = point["image_size"]
        in_channels_1 = point["in_channels"]
        kernel_size_1 = point["kernel_size"]

        batch_size_2 = point["batch_size"]+ 1
        image_size_2 = point["image_size"]+ 1
        in_channels_2 = point["in_channels"]+ 1
        kernel_size_2 = point["kernel_size"]+ 1

        batch_size_3 = point["batch_size"]+ 2
        image_size_3 = point["image_size"]+ 2
        in_channels_3 = point["in_channels"]+ 2
        kernel_size_3 = point["kernel_size"]+ 2

        batch_size_4 = point["batch_size"]+ 3 
        image_size_4 = point["image_size"]+ 3
        in_channels_4 = point["in_channels"]+ 3
        kernel_size_4 = point["kernel_size"]+ 3

        batch_size_5 = point["batch_size"]+ 4 
        image_size_5 = point["image_size"]+ 4
        in_channels_5 = point["in_channels"]+ 4
        kernel_size_5 = point["kernel_size"]+ 4

        print(point)

        import torch

        device, dtype = load_cuda_vs_knl(point)

        inputs_1 = torch.arange(
            batch_size_1 * image_size_1 * image_size_1 * in_channels_1,
            dtype=dtype,
            device=device,
        ).view((batch_size_1, in_channels_1, image_size_1, image_size_1))

        inputs_2 = torch.arange(
            batch_size_2 * image_size_2 * image_size_2 * in_channels_2,
            dtype=dtype,
            device=device,
        ).view((batch_size_2, in_channels_2, image_size_2, image_size_2))

        inputs_3 = torch.arange(
            batch_size_3 * image_size_3 * image_size_3 * in_channels_3,
            dtype=dtype,
            device=device,
        ).view((batch_size_3, in_channels_3, image_size_3, image_size_3))

        inputs_4 = torch.arange(
            batch_size_4 * image_size_4 * image_size_4 * in_channels_4,
            dtype=dtype,
            device=device,
        ).view((batch_size_4, in_channels_4, image_size_4, image_size_4))

        inputs_5 = torch.arange(
            batch_size_5 * image_size_5 * image_size_5 * in_channels_5,
            dtype=dtype,
            device=device,
        ).view((batch_size_5, in_channels_5, image_size_5, image_size_5))

        layer_1 = torch.nn.Conv2d(in_channels_1, out_channels, kernel_size_2, stride=1).to(device, dtype=dtype)
        layer_2 = torch.nn.Conv2d(in_channels_2, out_channels, kernel_size_2, stride=1).to(device, dtype=dtype)
        layer_3 = torch.nn.Conv2d(in_channels_3, out_channels, kernel_size_3, stride=1).to(device, dtype=dtype)
        layer_4 = torch.nn.Conv2d(in_channels_4, out_channels, kernel_size_4, stride=1).to(device, dtype=dtype)
        layer_5 = torch.nn.Conv2d(in_channels_5, out_channels, kernel_size_5, stride=1).to(device, dtype=dtype)

        #TODO
        ave_time_1 = benchmark_forward(layer_1, inputs_1)
        ave_time_2 = benchmark_forward(layer_2, inputs_2)
        ave_time_3 = benchmark_forward(layer_3, inputs_3)
        ave_time_4 = benchmark_forward(layer_4, inputs_4)
        ave_time_5 = benchmark_forward(layer_5, inputs_5)

        outputs_1 = layer_1(inputs_1)
        outputs_2 = layer_2(inputs_2)
        outputs_3 = layer_3(inputs_3)
        outputs_4 = layer_4(inputs_4)
        outputs_5 = layer_5(inputs_5)

        total_flop_1 = (
            kernel_size_1
            * kernel_size_1
            * in_channels_1
            * out_channels
            * outputs_1.shape[-1]
            * outputs_1.shape[-2]
            * batch_size_1)

        total_flop_2 = (
            kernel_size_2
            * kernel_size_2
            * in_channels_2
            * out_channels
            * outputs_2.shape[-1]
            * outputs_2.shape[-2]
            * batch_size_2)

        total_flop_3 = (
            kernel_size_3
            * kernel_size_3
            * in_channels_3
            * out_channels
            * outputs_3.shape[-1]
            * outputs_3.shape[-2]
            * batch_size_3)

        total_flop_4 = (
            kernel_size_4
            * kernel_size_4
            * in_channels_4
            * out_channels
            * outputs_4.shape[-1]
            * outputs_4.shape[-2]
            * batch_size_4)

        total_flop_5 = (
            kernel_size_5
            * kernel_size_5
            * in_channels_5
            * out_channels
            * outputs_5.shape[-1]
            * outputs_5.shape[-2]
            * batch_size_5)

        print("OUTPUT SHAPES: 1,2,3,4,5 (respectively)")
        print(outputs_1.shape)
        print(outputs_2.shape)
        print(outputs_3.shape)
        print(outputs_4.shape)
        print(outputs_5.shape)

        #TODO
        print("flop_1 = ", total_flop_1, "ave_time_1 = ", ave_time_1)
        ave_flops_1 = total_flop_1 / ave_time_1 * batch_size_1
        runtime_1 = time.time() - start
        print("runtime_1=", runtime_1, "ave_flops_1=", ave_flops_1)

        print("flop_2 = ", total_flop_2, "ave_time_2 = ", ave_time_2)
        ave_flops_2 = total_flop_2 / ave_time_2 * batch_size_2
        runtime_2 = time.time() - start
        print("runtime_2=", runtime_2, "ave_flops_2=", ave_flops_2)

        print("flop_3 = ", total_flop_3, "ave_time_3 = ", ave_time_3)
        ave_flops_3 = total_flop_3 / ave_time_3 * batch_size_3
        runtime_3 = time.time() - start
        print("runtime_3=", runtime_3, "ave_flops_3=", ave_flops_3)

        print("flop_4 = ", total_flop_4, "ave_time_4 = ", ave_time_4)
        ave_flops_4 = total_flop_4 / ave_time_4 * batch_size_4
        runtime_4 = time.time() - start
        print("runtime_4=", runtime_4, "ave_flops_4=", ave_flops_4)

        print("flop_5 = ", total_flop_5, "ave_time_5 = ", ave_time_5)
        ave_flops_5 = total_flop_5 / ave_time_5 * batch_size_5
        runtime_5 = time.time() - start
        print("runtime_5=", runtime_5, "ave_flops_5=", ave_flops_5)

        #TODO return tuple 
        return ave_flops_1

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
