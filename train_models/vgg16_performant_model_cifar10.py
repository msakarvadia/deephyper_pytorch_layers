import torch
import torchvision
from torch import nn
from typing import Union, List, Dict, Any, cast


class net(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True
    ) -> None:
        super(net, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(423 * 2 * 2, 13613),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(13613, 638),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(638, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=4, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'VGG16': [64, 64, 'M', 199, 199, 'M', 384, 384, 384, 'M', 765, 765, 765, 'M', 423, 423, 423, 'M'],
}


print("Creating Model")
VGG16_model = net(make_layers(cfgs['VGG16'], batch_norm=True))

print("Saving model")
# Additional information
PATH = "/home/mansisak/deephyper_pytorch_layers/vgg16/vgg16_models/performant_models/cifar10/vgg16_performant_model_0_epoch.pt"
torch.save({'model_state_dict': VGG16_model.state_dict()}, PATH)

print("Model saved")

def get_model():
    return VGG16_model

def get_batch_size():
    return 127
