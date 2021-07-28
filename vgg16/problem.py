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
Problem.add_dim("num_classes", [10])
Problem.add_dim("batch_size", (1, 128))
Problem.add_dim("image_size", [224])
Problem.add_dim("conv1_in_chan", [3])
Problem.add_dim("conv1_out_chan", (50, 200))
Problem.add_dim("conv_kern", (2, 4))
Problem.add_dim("pool_size", (2, 4))
Problem.add_dim("conv2_out_chan", (50, 200))
Problem.add_dim("conv3_out_chan", (100, 400))
Problem.add_dim("conv4_out_chan", (400, 800))
Problem.add_dim("conv5_out_chan", (400, 800))
Problem.add_dim("adaptive_pool_dim", (2, 8))
Problem.add_dim("fc1_out", (64, 16384))
Problem.add_dim("fc2_out", (32, 16384))

if use_knl:
    # Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_dim("omp_num_threads", [64])
    Problem.add_starting_point(
        num_classes=10,
        batch_size=128,
        image_size=224,
        conv1_in_chan=3,
        conv1_out_chan=64,
        conv_kern=3,
        pool_size=2,
        conv2_out_chan=128,
        conv3_out_chan=256,
        conv4_out_chan=512,
        conv5_out_chan=512,
        adaptive_pool_dim=7,
        fc1_out=4096,
        fc2_out=4096,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        num_classes=10,
        batch_size=128,
        image_size=224,
        conv1_in_chan=3,
        conv1_out_chan=64,
        conv_kern=3,
        pool_size=2,
        conv2_out_chan=128,
        conv3_out_chan=256,
        conv4_out_chan=512,
        conv5_out_chan=512,
        adaptive_pool_dim=7,
        fc1_out=4096,
        fc2_out=4096,
    )

if __name__ == "__main__":
    print(Problem)
