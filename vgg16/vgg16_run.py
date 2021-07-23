import time
import os
import sys
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa

#Portions of this VGG16 implementation come from:
#https://github.com/msyim/VGG16/blob/master/VGG16.py

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

def run(point):
    start = time.time()
    try:
        num_classes = point["num_classes"]
        batch_size = point["batch_size"]
        image_size = point["image_size"]
        conv1_in_chan = point["conv1_in_chan"]
        conv1_out_chan = point["conv1_out_chan"]
        conv_kern = point["conv_kern"]
        pool_size = point["pool_size"]
        conv2_out_chan = point["conv2_out_chan"]
        conv3_out_chan = point["conv3_out_chan"]
        conv4_out_chan = point["conv4_out_chan"]
        conv5_out_chan = point["conv5_out_chan"]
        adaptive_pool_dim = point["adaptive_pool_dim"]
        fc1_out = point["fc1_out"]
        fc2_out = point["fc2_out"]
        print(point)

        device, dtype = load_cuda_vs_knl(point)

        class VGG(nn.Module):

            def __init__(
                self,
                features,
                num_classes,
                batch_size,
                image_size,
                conv1_in_chan,
                conv1_out_chan,
                conv2_out_chan,
                conv3_out_chan, 
                conv4_out_chan,
                conv5_out_chan,
                conv_kern,
                pool_size,
                adaptive_pool_dim,
                fc1_out,
                fc2_out,
            ) -> None:
                super(VGG, self).__init__()
                self.flop = 0
                self.features = features
                self.avgpool = nn.AdaptiveAvgPool2d((adaptive_pool_dim, adaptive_pool_dim))
                self.classifier = nn.Sequential(
                    nn.Linear(conv5_out_chan * adaptive_pool_dim * adaptive_pool_dim, fc1_out),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(fc1_out, fc2_out),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(fc2_out, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
            layers: List[nn.Module] = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=pool_size, stride=2)]
                else:
                    v = cast(int, v)

                    conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kern, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        cfgs: Dict[str, List[Union[str, int]]] = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [conv1_out_chan, conv1_out_chan, 'M', 
                      conv2_out_chan, conv2_out_chan, 'M',
                      conv3_out_chan, conv3_out_chan, conv3_out_chan, 'M',
                      conv4_out_chan, conv4_out_chan, conv4_out_chan, 'M',
                      conv5_out_chan, conv5_out_chan, conv5_out_chan, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        inputs = torch.arange(
            batch_size * image_size ** 2 * conv1_in_chan, dtype=dtype, device=device
        ).view((batch_size, conv1_in_chan, image_size, image_size))

        #create and move model to GPU

        #"verion D" is VGG-16
        net = VGG(make_layers(cfgs['VGG16'], batch_norm=True),
                num_classes,
                batch_size,
                image_size,
                conv1_in_chan,
                conv1_out_chan,
                conv2_out_chan,
                conv3_out_chan, 
                conv4_out_chan,
                conv5_out_chan,
                conv_kern,
                pool_size,
                adaptive_pool_dim,
                fc1_out,
                fc2_out,
                ).to(device, dtype=dtype)         

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
        "num_classes": 100,
        "batch_size": 128,
        "image_size": 224,
        "conv1_in_chan": 3,
        "conv1_out_chan": 64,
        "conv2_out_chan": 128,
        "conv3_out_chan": 256,
        "conv4_out_chan": 512,
        "conv5_out_chan": 512,
        "conv_kern": 3,
        "pool_size": 2,
        "adaptive_pool_dim": 7,
        "fc1_out": 4096,
        "fc2_out": 4096,
    }

    if use_knl:
        point["omp_num_threads"] = 64

    print("flops for this setting =", run(point))
