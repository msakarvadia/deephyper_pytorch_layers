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
        parser.add_argument('--load_model_cp', type=str, default="model.pt", help='Full path to model *.pt checkpoint file')
        parser.add_argument('--save_model', type=str, default="model.pt", help='Full path to desired location for model *.pt checkpoint file')
        parser.add_argument('--model_class_file', type=str, default="model.py", help='Name of file which defines model class (needs to be stored somewhere in this git repo)')
        
    args = parser.parse_args()

    #load model from class
    from args.model_class_file import model
    model = model()

    #load model from checkpoint

    #load dataset

    #train model and check accuracy on validation data after each epoch

    #save model to checkpoint

    
    # Additional information
