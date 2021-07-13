from deephyper.benchmark import HpProblem
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import use_knl  # noqa

Problem = HpProblem()
Problem.add_dim("batch_size", (1, 64))
Problem.add_dim("image_size", [224])
Problem.add_dim("conv1_in_chan", [3])
Problem.add_dim("conv1_out_chan", (50, 150))
Problem.add_dim("conv1_kern", (2, 23))
Problem.add_dim("pool_size_1", (2, 11)
Problem.add_dim("pool_size_2", (2, 11))
Problem.add_dim("pool_size_5", (2, 11))
Problem.add_dim("conv2_out_chan", (200, 350))
Problem.add_dim("conv2_kern", (3, 8))
Problem.add_dim("conv3_out_chan", (300, 450))
Problem.add_dim("conv3_kern", (3, 8))
Problem.add_dim("conv4_out_chan", (300, 450))
Problem.add_dim("conv4_kern", (3, 8))
Problem.add_dim("conv5_out_chan", (200, 350))
Problem.add_dim("conv5_kern", (3, 8))
Problem.add_dim("adaptive_pool_dim", (3, 8))
Problem.add_dim("fc1_out", (64, 16384))
Problem.add_dim("fc2_out", (32, 16384))
Problem.add_dim("fc3_out", [1000])

if use_knl:
    # Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_dim("omp_num_threads", [64])
    Problem.add_starting_point(
        batch_size=10,
        image_size=32,
        conv1_in_chan=3,
        conv1_out_chan=16,
        conv1_kern=5,
        pool_size_1=2,
        pool_size_2=2,
        pool_size_5=2,
        conv2_out_chan=16,
        conv2_kern=5,
        conv3_out_chan=16,
        conv3_kern=5,
        conv4_out_chan=16,
        conv4_kern=5,
        conv5_out_chan=16,
        conv5_kern=5,
        adaptive_pool_dim=5,
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
        pool_size_1=2,
        pool_size_2=2,
        pool_size_5=2,
        conv2_out_chan=16,
        conv2_kern=5,
        conv3_out_chan=16,
        conv3_kern=5,
        conv4_out_chan=16,
        conv4_kern=5,
        conv5_out_chan=16,
        conv5_kern=5,
        adaptive_pool_dim=5,
        fc1_out=128,
        fc2_out=84,
        fc3_out=10,
    )

if __name__ == "__main__":
    print(Problem)
