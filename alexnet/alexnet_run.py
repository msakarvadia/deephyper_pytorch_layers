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
        conv3_out_chan = point["conv3_out_chan"]
        conv3_kern = point["conv3_kern"]
        conv4_out_chan = point["conv4_out_chan"]
        conv4_kern = point["conv4_kern"]
        conv5_out_chan = point["conv5_out_chan"]
        conv5_kern = point["conv5_kern"]
        adaptive_pool_dim = point["adaptive_pool_dim"]
        fc1_out = point["fc1_out"]
        fc2_out = point["fc2_out"]
        fc3_out = point["fc3_out"]
        print(point)
        import torch
        import torch.nn as nn

        device, dtype = load_cuda_vs_knl(point)

        class AlexNet(nn.Module):

            def __init__( 
                self,
                batch_size,
                image_size,
                conv1_in_chan,
                conv1_out_chan,
                conv1_kern,
                pool_size,
                conv2_out_chan,
                conv2_kern,
                conv3_out_chan,
                conv3_kern,
                conv4_out_chan,
                conv4_kern,
                conv5_out_chan,
                conv5_kern,
                adaptive_pool_dim,
                fc1_out,
                fc2_out,
                fc3_out
            ):
                super(AlexNet, self).__init__()
                #TODO replace all hyper parameters with variable
                self.flop = 0
                self.features = nn.Sequential(
                    #1st conv
                    nn.Conv2d(conv1_in_chan, conv1_out_chan, kernel_size=conv1_kern, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size, stride=2),
                    #2nd conv
                    nn.Conv2d(conv1_out_chan, conv2_out_chan, kernel_size=conv2_kern, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size, stride=2),
                    #3rd conv
                    nn.Conv2d(conv2_out_chan, conv3_out_chan, kernel_size=conv3_kern, padding=1),
                    nn.ReLU(inplace=True),
                    #4th conv
                    nn.Conv2d(conv3_out_chan, conv4_out_chan, kernel_size=conv4_kern, padding=1),
                    nn.ReLU(inplace=True),
                    #5th conv
                    nn.Conv2d(conv4_out_chan, conv5_out_chan, kernel_size=conv5_kern, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size, stride=2),
                )

                self.avgpool = nn.AdaptiveAvgPool2d((adaptive_pool_dim, adaptive_pool_dim))

                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(conv5_out_chan * adaptive_pool_dim**2, fc1_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(fc1_out, fc2_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(fc2_out, fc3_out),
                )

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

        #create and move model to GPU
        net = AlexNet(
                batch_size,
                image_size,
                conv1_in_chan,
                conv1_out_chan,
                conv1_kern,
                pool_size,
                conv2_out_chan,
                conv2_kern,
                conv3_out_chan,
                conv3_kern,
                conv4_out_chan,
                conv4_kern,
                conv5_out_chan,
                conv5_kern,
                adaptive_pool_dim,
                fc1_out,
                fc2_out,
                fc3_out).to(device, dtype=dtype)

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
        "conv3_out_chan": 128,
        "conv3_kern": 4,
        "conv4_out_chan": 256,
        "conv4_kern": 5,
        "conv5_out_chan": 256,
        "conv5_kern": 2,
        "adaptive_pool_dim": 6,
        "fc1_out": 15545,
        "fc2_out": 15002,
        "fc3_out": 10,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
