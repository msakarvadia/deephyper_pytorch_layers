import time


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        image_size = point["image_size"]
        in_channels = point["in_channels"]
        out_channels = point["out_channels"]
        kernel_size = point["kernel_size"]
        # omp_num_threads = point['omp_num_threads']

        import os

        # os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
        # os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)
        # os.environ['KMP_HW_SUBSET'] = '1s,%sc,2t' % str(omp_num_threads)
        # os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
        # os.environ['KMP_BLOCKTIME'] = str(0)
        # os.environ['MKLDNN_VERBOSE'] = str(1)
        import torch

        print("torch version: ", torch.__version__, " torch file: ", torch.__file__)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        inputs = torch.arange(
            batch_size * image_size ** 3 * in_channels, dtype=torch.float, device=device
        ).view((batch_size, in_channels, image_size, image_size, image_size))

        layer = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        ).to(device, dtype=torch.float32)
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

        runs = 5
        tot_time = 0.0
        tt = time.time()
        for _ in range(runs):
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

        print("received exception: ", str(e), "for point: ", point)
        print(traceback.print_exc())
        print("runtime=", time.time() - start)
        return 0.0


if __name__ == "__main__":
    point = {
        "batch_size": 10,
        "image_size": 128,
        "in_channels": 3,
        "out_channels": 3,
        "kernel_size": 4,
        # 'omp_num_threads':64,
    }

    print("flops for this setting =", run(point))
