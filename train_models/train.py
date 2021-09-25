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
    #TODO add arge parse option for different datasets


    #train model and check accuracy on validation data after each epoch
    #TODO GRANULAR TIMING BREAK DOWN

    #save model to checkpoint
    
#set arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train custom model on CIFAR100, Fruits, or ImageNet Dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to Train on')
    parser.add_argument('--load_model_cp', type=str, default="model.pt", help='Full path to model *.pt checkpoint file')
    parser.add_argument('--save_model_cp', type=str, default="model.pt", help='Full path to desired location for model *.pt checkpoint file')
    parser.add_argument('--network_class_file', type=str, default="model.py", help='Name of file which defines model class - class must be named: net() (file needs to be stored somewhere in this git repo)')
    

    
args = parser.parse_args()
main(args)
