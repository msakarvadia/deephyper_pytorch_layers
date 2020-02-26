from deephyper.benchmark import HpProblem
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_torch import use_knl  # noqa


Problem = HpProblem()
Problem.add_dim("batch_size", (1, 512))
Problem.add_dim("height", (128, 1024))
Problem.add_dim("width", (128, 1024))
Problem.add_dim("in_channels", (2, 64))
Problem.add_dim("out_channels", (2, 64))
Problem.add_dim("kernel_size", (2, 16))

if use_knl:
    # KGF: unlike 1D and 3D conv*/ problems, not restricted to 64 threads:
    Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_starting_point(
        batch_size=128,
        height=512,
        width=512,
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        omp_num_threads=64,
    )
else:
    Problem.add_starting_point(
        batch_size=128,
        height=512,
        width=512,
        in_channels=3,
        out_channels=64,
        kernel_size=3,
    )


if __name__ == "__main__":
    print(Problem)
