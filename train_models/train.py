import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import argparse, sys
import importlib
sys.path.append('../..')

from tools.CalcMean import CalcMean

def main(args):
    #load model from class
    model_class = importlib.import_module(args.network_class_file)
    model = model_class.get_model()
    batch_size = model_class.get_batch_size()
    print("loaded model architecture")

    #load model from checkpoint
    checkpoint = torch.load(args.load_model_cp)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("loaded model state from checkpoint")
    
    #Tell the model it is in training mode:
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Verifying CUDA
    print("Device: ",device)

    #Model to device
    model.to(device)

    #Define optimizer+criterion
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    #load dataset

    transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.data == "cifar10":
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        print("loaded cifar10")

    if args.data == "fruits":
        train_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Training', transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/fruits-360/Test', transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        print("loaded fruits")

    if args.data == "imagenet":
        train_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/imagenet/train', transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        test_data = torchvision.datasets.ImageFolder(root='/home/felker/resnet-ASP/imagenet/val', transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        print("loaded imagenet")

    #define accuracy funcitons
    avg_fwd_pass_for_acc_time = CalcMean()
    def accuracy(model):
        #Testing Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)

                #Check Time on forward pass
                torch.cuda.synchronize()
                start_fwd_acc = time.time()
                outputs = model(images)
                torch.cuda.synchronize()
                end_fwd_acc = time.time()
                avg_fwd_pass_for_acc_time.add_value(end_fwd_acc - start_fwd_acc)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
        return 100 * correct / total


    #train model and check accuracy on validation data after each epoch
    torch.cuda.synchronize()
    start = time.time()
    epochs = []
    acc = []
    avg_acc_time = CalcMean()
    avg_input_time = CalcMean()
    avg_fwd_pass_time = CalcMean()
    avg_back_pass_time = CalcMean()
    avg_epoch_training_time = CalcMean()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        epochs.append(epoch)

        #Check accuracy
        torch.cuda.synchronize()
        start_acc = time.time()

        acc.append(accuracy(model))

        torch.cuda.synchronize()
        end_acc = time.time()
        avg_acc_time.add_value(end_acc - start_acc)

        print(epochs)
        print(acc)
        running_loss = 0.0

        #Time to train one epoch
        torch.cuda.synchronize()
        start_epoch = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            torch.cuda.synchronize()
            start_input = time.time()

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            torch.cuda.synchronize()
            end_input = time.time()
            avg_input_time.add_value(end_input - start_input)

            # forward pass
            torch.cuda.synchronize()
            start_fwd = time.time()

            output = model(inputs)
            loss = criterion(output, labels)

            torch.cuda.synchronize()
            end_fwd = time.time()
            avg_fwd_pass_time.add_value(end_fwd - start_fwd)

            # backward pass 
            torch.cuda.synchronize()
            start_bwd = time.time()

            #update weights
            loss.backward()

            #updated weights
            optimizer.step()

            torch.cuda.synchronize()
            end_bwd = time.time()
            avg_back_pass_time.add_value(end_bwd - start_bwd)

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        torch.cuda.synchronize()
        end_epoch = time.time()
        avg_epoch_training_time.add_value(end_epoch - start_epoch)

    torch.cuda.synchronize()
    end = time.time()
    print("Time to train model: ", end - start)
    print("Average time for calculating accuracy over whole val set: ", avg_acc_time.mean())
    print("Average time for forward pass over batch while checking accuracy: ", avg_fwd_pass_for_acc_time.mean())
    print("Average time for training one full epoch: ", avg_epoch_training_time.mean())
    print("Average time for loading input data during training: ", avg_input_time.mean())
    print("Average time for forward pass over batch during training: ", avg_fwd_pass_time.mean())
    print("Average time for backward pass over batch during training: ", avg_back_pass_time.mean())
    print("epoch and accuracy during training: ")
    print(epochs)
    print(acc)
    print("Final Accuracy: ", accuracy(model))

    #save model to checkpoint
    PATH = args.save_model_cp

    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)
    
#set arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train custom model on CIFAR100, Fruits, or ImageNet Dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to Train on')
    parser.add_argument('--load_model_cp', type=str, help='Full path to model *.pt checkpoint file', required=True)
    parser.add_argument('--save_model_cp', type=str, help='Full path to desired location for model *.pt checkpoint file', required=True)
    parser.add_argument('--network_class_file', type=str, help='Name of file which defines model class, omit the *.py ending - class must be named: net() (file must be stored this repo)', required=True)
    parser.add_argument('--data', type=str, default="cifar10", help='Choose dataset: cifar10, fruits, imagenet')
    

    
args = parser.parse_args()
main(args)
