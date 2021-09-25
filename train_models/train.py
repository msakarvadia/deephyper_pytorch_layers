import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader

def main(args):

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Train custom model on CIFAR100, Fruits, or ImageNet Dataset')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
        parser.add_argument('--epochs', type=int, default=1, help='Epochs to Train on')
        parser.add_argument('--load_model', type=str, default="model.pt", help='Full path to model *.pt file')
        parser.add_argument('--save_model', type=str, default="model.pt", help='Full path to desired location for model *.pt file')
        
