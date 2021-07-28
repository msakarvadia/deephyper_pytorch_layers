import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Dict, Any, cast


transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Downloading training data
#train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Training', transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

#Downloading test data
#test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Test', transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()

#Now using VGG16
class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 131,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
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
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

VGG16_model = VGG(make_layers(cfgs['VGG16'], batch_norm=True))

#Model description
VGG16_model.eval()

#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Verifying CUDA
print(device)

#Move the input and VGG16_model to GPU for speed if available
VGG16_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.001, momentum=0.9) #77% - for CIFAR10
#optimizer = optim.Adam(VGG16_model.parameters(), lr=0.001) #only predicted one class
#optimizer = optim.SGD(VGG16_model.parameters(), lr=0.002, momentum=0.9) #64

def accuracy():
    #Testing Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = VGG16_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return 100 * correct / total

torch.cuda.synchronize()
start = time.time()
epochs = []
acc = []
for epoch in range(10):  # loop over the dataset multiple times
    epochs.append(epoch)
    acc.append(accuracy())
    print(epochs)
    print(acc)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = VGG16_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
torch.cuda.synchronize()
end = time.time()
print("Time to train model: ", end - start)
print("epoch and accuracy during training: ")
print(epochs)
print(acc)
print("Final Accuracy: ", accuracy())
accuracy()
