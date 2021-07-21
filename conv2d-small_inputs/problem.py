from deephyper.benchmark import HpProblem

# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import use_knl  # noqa

Problem = HpProblem()
Problem.add_dim("batch_size", (1, 1024))
Problem.add_dim("image_size", (32, 64))
Problem.add_dim("in_channels", (2, 8))
Problem.add_dim("out_channels", (2, 16))
Problem.add_dim("kernel_size", (2, 4))

if use_knl:
    # KGF: unlike 1D and 3D conv*/ problems, not restricted to 64 threads:
    Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_starting_point(
        batch_size=128,
        image_size=32,
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        batch_size=128, image_size=32, in_channels=2, out_channels=2, kernel_size=2
    )

if __name__ == "__main__":
    print(Problem)
