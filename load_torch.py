import os

use_knl = False
# TODO(KGF): more robust way to detect presence of an Intel KNL device?
# this might be unset for the default affinity? Check hostname for Theta instead?
if os.environ.get("KMP_AFFINITY") is not None:
    use_knl = True


def cuda_vs_knl(point):
    import torch

    print("torch version: ", torch.__version__, " torch file: ", torch.__file__)
    use_cuda = torch.cuda.is_available()
    print("PyTorch: CUDA available? {}".format(use_cuda))
    if use_cuda:
        assert not use_knl
        print("torch.cuda.current_device() = {}".format(torch.cuda.current_device()))
        print("CUDA_DEVICE_ORDER = {}".format(os.environ["CUDA_DEVICE_ORDER"]))
        print("CUDA_VISIBLE_DEVICES = {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
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
        assert use_knl
        # TODO(KGF): assuming KNL if not CUDA GPU; add KNL vs. "regular CPU" switch
        omp_num_threads = point["omp_num_threads"]
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_num_threads)
        os.environ["KMP_HW_SUBSET"] = "1s,%sc,2t" % str(omp_num_threads)
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
