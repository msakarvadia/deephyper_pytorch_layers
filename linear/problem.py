from deephyper.benchmark import HpProblem
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_torch import use_knl  # noqa

Problem = HpProblem()
Problem.add_dim("batch_size", (1, 8192))
Problem.add_dim("in_features", (128, 8192))
Problem.add_dim("out_features", (128, 8192))
Problem.add_dim("bias", [0, 1])

if use_knl:
    Problem.add_dim("omp_num_threads", (8, 64))
    Problem.add_starting_point(
        batch_size=128, in_features=1024, out_features=512, bias=0, omp_num_threads=64
    )
else:
    Problem.add_starting_point(
        batch_size=128, in_features=1024, out_features=512, bias=0
    )


if __name__ == "__main__":
    print(Problem)
