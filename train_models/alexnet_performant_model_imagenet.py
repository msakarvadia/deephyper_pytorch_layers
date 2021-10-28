import torch
import torchvision
from torch import nn


#Now using the AlexNet
class net(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 127, kernel_size=10, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(127, 207, kernel_size=8, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(207, 324, kernel_size=8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(324, 356, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(356, 270, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(270 * 3 * 3, 5485),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(5485, 102),
            nn.ReLU(inplace=True),
            nn.Linear(102, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

print("Creating model")
AlexNet_model = net()

print("Saving model")
# Additional information
PATH = "/home/mansisak/deephyper_pytorch_layers/alexnet/alexnet_models/performant_models/imagenet/alexnet_performant_model_0_epoch.pt"
torch.save({'model_state_dict': AlexNet_model.state_dict()}, PATH)

print("Model saved")

def get_batch_size():
    return 124

def get_model():
    return AlexNet_model
