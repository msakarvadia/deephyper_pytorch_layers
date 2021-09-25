import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import argparse
import importlib

def main(args):
    #load model from class
    model_class = importlib.import_module(args.network_class_file)
    model = model_class.net()
    print("loaded model architecture")

    #load model from checkpoint
    checkpoint = torch.load(args.load_model_cp)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("loaded model state from checkpoint")

    #Tell the model it is in training mode:
    model.train()

    #Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #load dataset

    transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    ])

    if args.data == "cifar100":
        train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 100
        print("loaded cifar100")

    if args.data == "fruits":
        train_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Training', transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Test', transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 131
        print("loaded fruits")

    if args.data == "imagenet":
        train_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/imagenet/train', transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/imagenet/val', transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 1000
        print("loaded imagenet")

    #define accuracy funcitons
    def accuracy():
        #Testing Accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
        return 100 * correct / total


    #train model and check accuracy on validation data after each epoch
    #TODO GRANULAR TIMING BREAK DOWN

    #save model to checkpoint
    
#set arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train custom model on CIFAR100, Fruits, or ImageNet Dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to Train on')
    parser.add_argument('--batch_size', type=int, default=128, help='Epochs to Train on')
    parser.add_argument('--load_model_cp', type=str, default="model.pt", help='Full path to model *.pt checkpoint file')
    parser.add_argument('--save_model_cp', type=str, default="model.pt", help='Full path to desired location for model *.pt checkpoint file')
    parser.add_argument('--network_class_file', type=str, default="model.py", help='Name of file which defines model class - class must be named: net() (file needs to be stored somewhere in this git repo)')
    parser.add_argument('--data', type=str, default="cifar100", help='Choose dataset: cifar100, fruits, imagenet')
    

    
args = parser.parse_args()
main(args)
