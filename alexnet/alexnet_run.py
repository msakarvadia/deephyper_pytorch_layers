import time
import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa


def run(point):
    start = time.time()
    try:
        batch_size = point["batch_size"]
        image_size = point["image_size"]
        conv1_in_chan = point["conv1_in_chan"]
        conv1_out_chan = point["conv1_out_chan"]
        conv1_kern = point["conv1_kern"]
        pool_size = point["pool_size"]
        conv2_out_chan = point["conv2_out_chan"]
        conv2_kern = point["conv2_kern"]
        fc1_out = point["fc1_out"]
        fc2_out = point["fc2_out"]
        fc3_out = point["fc3_out"]
        print(point)
        import torch
        import torch.nn as nn

        device, dtype = load_cuda_vs_knl(point)

        class AlexNet(nn.Module):

            def __init__(self, num_classes: int = 1000) -> None:
                super(AlexNet, self).__init__()
                self.flop = 0
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ).to(device, dtype=dtype)

                self.avgpool = nn.AdaptiveAvgPool2d((6, 6)).to(device, dtype=dtype)

                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                ).to(device, dtype=dtype)

                #TODO update flop calculations:
                self.flop += 1000

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        inputs = torch.arange(
            batch_size * image_size ** 2 * conv1_in_chan, dtype=dtype, device=device
        ).view((batch_size, conv1_in_chan, image_size, image_size))

        net = AlexNet()

        total_flop = net.flop

        ave_time = benchmark_forward(net, inputs)

        print("total_flop = ", total_flop, "ave_time = ", ave_time)

        ave_flops = total_flop / ave_time
        runtime = time.time() - start
        print("runtime=", runtime, "ave_flops=", ave_flops)

        return ave_flops
    except Exception as e:
        import traceback

        print("received exception: ", str(e), "for point: ", point)
        print(traceback.print_exc())
        print("runtime=", time.time() - start)
        return 0.0


if __name__ == "__main__":
    point = {
        "batch_size": 23,
        "image_size": 64,
        "conv1_in_chan": 3,
        "conv1_out_chan": 54,
        "conv1_kern": 6,
        "pool_size": 2,
        "conv2_out_chan": 56,
        "conv2_kern": 4,
        "fc1_out": 15545,
        "fc2_out": 15002,
        "fc3_out": 10,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
