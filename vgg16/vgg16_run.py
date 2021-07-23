import time
import os
import sys
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl  # noqa

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

                #FLOPS calculations for convolutional layers:
                layer_input_size = image_size
                #1st block of convolutional layers
                for i in range(2):
                    if i == 1:
                        self.flop += (
                        conv_kern ** 2
                        * conv1_in_chan
                        * conv1_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    else:
                        self.flop += (
                        conv_kern ** 2
                        * conv1_out_chan
                        * conv1_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
    
                    layer_input_size = int(((layer_input_size - conv_kern + 2 * 1) / 1) + 1) 
    
                #Reshape for max pool layer:
                layer_input_size = int(((layer_input_size - pool_size) / 2) + 1)

                #2nd block of convolutional layers
                for i in range(2):
                    if i == 1:
                        self.flop += (
                        conv_kern ** 2
                        * conv1_out_chan
                        * conv2_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    else:
                        self.flop += (
                        conv_kern ** 2
                        * conv2_out_chan
                        * conv2_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    layer_input_size = int(((layer_input_size - conv_kern + 2 * 1) / 1) + 1) 
    
                #Reshape for max pool layer:
                layer_input_size = int(((layer_input_size - pool_size) / 2) + 1)

                #3rd block of convolutional layers
                for i in range(3):
                    if i == 1:
                        self.flop += (
                        conv_kern ** 2
                        * conv2_out_chan
                        * conv3_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    else:
                        self.flop += (
                        conv_kern ** 2
                        * conv3_out_chan
                        * conv3_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    layer_input_size = int(((layer_input_size - conv_kern + 2 * 1) / 1) + 1) 
    
                #Reshape for max pool layer:
                layer_input_size = int(((layer_input_size - pool_size) / 2) + 1)

                #4th block of convolutional layers
                for i in range(3):
                    if i == 1:
                        self.flop += (
                        conv_kern ** 2
                        * conv3_out_chan
                        * conv4_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    else:
                        self.flop += (
                        conv_kern ** 2
                        * conv4_out_chan
                        * conv4_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    layer_input_size = int(((layer_input_size - conv_kern + 2 * 1) / 1) + 1) 
    
                #Reshape for max pool layer:
                layer_input_size = int(((layer_input_size - pool_size) / 2) + 1)

                #5th block of convolutional layers
                for i in range(3):
                    if i == 1:
                        self.flop += (
                        conv_kern ** 2
                        * conv4_out_chan
                        * conv5_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    else:
                        self.flop += (
                        conv_kern ** 2
                        * conv5_out_chan
                        * conv5_out_chan
                        * layer_input_size ** 2
                        * batch_size
                        )
                    layer_input_size = int(((layer_input_size - conv_kern + 2 * 1) / 1) + 1) 
                
                #Reshape for max pool layer:
                layer_input_size = int(((layer_input_size - pool_size) / 2) + 1)

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

                # FLOPS calculatios for linear layers
                # 1st linear layer
                self.flop += (
                    (2 * (conv5_out_chan * adaptive_pool_dim ** 2) - 1)
                    * fc1_out
                    * batch_size
                )

                # 2nd linear layer
                self.flop += (2 * fc1_out - 1) * fc2_out * batch_size

                # 3rd linear layer
                self.flop += (2 * fc2_out - 1) * num_classes * batch_size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
            layers: List[nn.Module] = []
            in_channels = conv1_in_chan 
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
            'VGG16': [conv1_out_chan, conv1_out_chan, 'M', 
                      conv2_out_chan, conv2_out_chan, 'M',
                      conv3_out_chan, conv3_out_chan, conv3_out_chan, 'M',
                      conv4_out_chan, conv4_out_chan, conv4_out_chan, 'M',
                      conv5_out_chan, conv5_out_chan, conv5_out_chan, 'M'],
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
