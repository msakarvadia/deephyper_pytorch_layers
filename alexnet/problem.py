from deephyper.benchmark import HpProblem
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import use_knl  # noqa

#NOTE(MS): I have intentially left the ranges for some of the hyper-parameters large in
# order to facilitate a large number of combinations. There will be configurations of 
#these ranges that cause errors related to kernels being larger than images, but I am 
#choosing to ignore those cases for now because limiting the ranges would greatly 
#reduce the valid combinations of hyper-parameters. Feel free to change ranges based 
#on your needs.

#TODO(MS): Implement conditional hyper-parameter ranges

Problem = HpProblem()
Problem.add_dim("batch_size", (1, 128))
Problem.add_dim("image_size", [224])
Problem.add_dim("conv1_in_chan", [3])
Problem.add_dim("conv1_out_chan", (50, 150))
Problem.add_dim("conv1_kern", (9, 13))
Problem.add_dim("pool_size_1", (2, 5))
Problem.add_dim("pool_size_2", (2, 5))
Problem.add_dim("pool_size_5", (2, 5))
Problem.add_dim("conv2_out_chan", (200, 350))
Problem.add_dim("conv2_kern", (2, 8))
Problem.add_dim("conv3_out_chan", (300, 450))
Problem.add_dim("conv3_kern", (2, 8))
Problem.add_dim("conv4_out_chan", (300, 450))
Problem.add_dim("conv4_kern", (2, 8))
Problem.add_dim("conv5_out_chan", (200, 350))
Problem.add_dim("conv5_kern", (2, 8))
Problem.add_dim("adaptive_pool_dim", (2, 8))
Problem.add_dim("fc1_out", (500, 6000))
Problem.add_dim("fc2_out", (500, 6000))
Problem.add_dim("fc3_out", [10])

if use_knl:
    # Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_dim("omp_num_threads", [64])
    Problem.add_starting_point(
        batch_size=10,
        image_size=224,
        conv1_in_chan=3,
        conv1_out_chan=96,
        conv1_kern=11,
        pool_size_1=3,
        pool_size_2=3,
        pool_size_5=3,
        conv2_out_chan=256,
        conv2_kern=5,
        conv3_out_chan=384,
        conv3_kern=3,
        conv4_out_chan=384,
        conv4_kern=3,
        conv5_out_chan=256,
        conv5_kern=3,
        adaptive_pool_dim=5,
        fc1_out=4096,
        fc2_out=1024,
        fc3_out=10,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        batch_size=10,
        image_size=224,
        conv1_in_chan=3,
        conv1_out_chan=96,
        conv1_kern=11,
        pool_size_1=3,
        pool_size_2=3,
        pool_size_5=3,
        conv2_out_chan=256,
        conv2_kern=5,
        conv3_out_chan=384,
        conv3_kern=3,
        conv4_out_chan=384,
        conv4_kern=3,
        conv5_out_chan=256,
        conv5_kern=3,
        adaptive_pool_dim=5,
        fc1_out=4096,
        fc2_out=1024,
        fc3_out=10,
    )

if __name__ == "__main__":
    print(Problem)
