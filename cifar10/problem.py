from deephyper.benchmark import HpProblem
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import use_knl  # noqa


Problem = HpProblem()
Problem.add_dim("batch_size", (1, 32))
Problem.add_dim("image_size", [32])
Problem.add_dim("conv1_in_chan", [3])
Problem.add_dim("conv1_out_chan", (3, 64))
Problem.add_dim("conv1_kern", (3, 8))
Problem.add_dim("pool_size", [2])
Problem.add_dim("conv2_out_chan", (3, 64))
Problem.add_dim("conv2_kern", (3, 8))
# Problem.add_dim("fc1_out", (64, 256))
# Problem.add_dim("fc2_out", (32, 128))
Problem.add_dim("fc1_out", (64, 16384))
Problem.add_dim("fc2_out", (32, 16384))
Problem.add_dim("fc3_out", [10])

if use_knl:
    # Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_dim("omp_num_threads", [64])
    Problem.add_starting_point(
        batch_size=10,
        image_size=32,
        conv1_in_chan=3,
        conv1_out_chan=16,
        conv1_kern=5,
        pool_size=2,
        conv2_out_chan=16,
        conv2_kern=5,
        fc1_out=128,
        fc2_out=84,
        fc3_out=10,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        batch_size=10,
        image_size=32,
        conv1_in_chan=3,
        conv1_out_chan=16,
        conv1_kern=5,
        pool_size=2,
        conv2_out_chan=16,
        conv2_kern=5,
        fc1_out=128,
        fc2_out=84,
        fc3_out=10,
    )

if __name__ == "__main__":
    print(Problem)
