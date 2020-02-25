import time
import os

# from ptflops import get_model_complexity_info


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        in_features = point["in_features"]
        out_features = point["out_features"]
        bias = int(point["bias"]) == 1

        import torch

        print("torch version: ", torch.__version__, " torch file: ", torch.__file__)
        use_cuda = torch.cuda.is_available()
        print("PyTorch: CUDA available? {}".format(use_cuda))
        if use_cuda:
            print(
                "torch.cuda.current_device() = {}".format(torch.cuda.current_device())
            )
            print("CUDA_DEVICE_ORDER = {}".format(os.environ["CUDA_DEVICE_ORDER"]))
            print(
                "CUDA_VISIBLE_DEVICES = {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
            )
            # print("torch.cuda.device(0) = {}".format(torch.cuda.device(0)))
            # print("torch.cuda.device_count() = {}".format(torch.cuda.device_count()))
            # print("torch.cuda.get_device_name(0) = {}".format(
            #     torch.cuda.get_device_name(0)))
            device = torch.device("cuda")
            # https://pytorch.org/docs/stable/tensors.html
            dtype = torch.float  # equivalent to torch.float32
            # dtype = torch.cuda.FloatTensor
            torch.backends.cudnn.benchmark = True
        else:
            # TODO(KGF): assuming KNL if not CUDA GPU; add KNL vs. "regular CPU" switch
            omp_num_threads = point["omp_num_threads"]
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
            os.environ["MKL_NUM_THREADS"] = str(omp_num_threads)
            os.environ["KMP_HW_SUBSET"] = "1s,%sc,2t" % str(omp_num_threads)
            os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
            os.environ["KMP_BLOCKTIME"] = str(0)
            os.environ["MKLDNN_VERBOSE"] = str(1)
            os.environ["MKL_VERBOSE"] = str(1)
            device = torch.device("cpu")
            dtype = torch.float32
            # dtype = torch.FloatTensor

        print(point)

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
        )  # dtype=float)
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
        # 'batch_size': 10,
        "in_features": 512,
        "out_features": 512,
        # KGF: large problem size for initial evlauation:
        # "batch_size": 128,
        # "in_features": 4096,
        # "out_features": 4096,
        "bias": 1,
        # 'omp_num_threads':64,
    }

    print("flops for this setting =", run(point))
