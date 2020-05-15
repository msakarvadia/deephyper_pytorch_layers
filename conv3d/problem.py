from deephyper.benchmark import HpProblem
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import use_knl  # noqa

Problem = HpProblem()

Problem.add_dim("batch_size", (1, 64))
Problem.add_dim("image_size", (16, 128))
Problem.add_dim("in_channels", (2, 64))
Problem.add_dim("out_channels", (2, 64))
Problem.add_dim("kernel_size", (2, 10))

if use_knl:
    Problem.add_dim("omp_num_threads", [64])
    # Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_starting_point(
        batch_size=10,
        image_size=28,
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        batch_size=10, image_size=28, in_channels=2, out_channels=2, kernel_size=2
    )

if __name__ == "__main__":
    print(Problem)
