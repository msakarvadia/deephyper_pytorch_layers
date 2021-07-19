import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.

    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3

    Returns
    -------
    gpu_memory_map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
    )
    # Python 3 compatibility:
    # https://stackoverflow.com/questions/49595663/find-a-gpu-with-enough-memory
    result = result.decode("utf-8")
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    return gpu_memory_map


def get_first_gpu_memory_usage():
    """Get the current gpu usage.

    Returns
    -------
    usage: int
        Memory usage as integer in MB.
    """

    mem = get_gpu_memory_map()

    return int(mem[0])
