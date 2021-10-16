import torch
import torchvision
from torch import nn


#Now using the AlexNet
class net(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=10, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 261, kernel_size=8, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(261, 444, kernel_size=8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(444, 427, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(427, 322, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(322 * 2 * 2, 663),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(663, 3582),
            nn.ReLU(inplace=True),
            nn.Linear(3582, num_classes),
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
PATH = "/home/mansisak/deephyper_pytorch_layers/alexnet/alexnet_models/performant_models/cifar10/alexnet_performant_model_0_epoch.pt"
torch.save({'model_state_dict': AlexNet_model.state_dict()}, PATH)

print("Model saved")

def get_batch_size():
    return 120

def get_model():
    return AlexNet_model
