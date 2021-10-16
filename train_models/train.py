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
    torch.cuda.synchronize()
    start = time.time()
    epochs = []
    acc = []
    fwd_pass_time = []
    back_pass_time = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times
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

            # forward pass
            torch.cuda.synchronize()
            start_fwd = time.time()

            output = model(inputs)
            loss = criterion(output, labels)

            torch.cuda.synchronize()
            end_fwd = time.time()
            fwd_pass_time.append(end_fwd - start_fwd)

            # backward pass 
            torch.cuda.synchronize()
            start_bwd = time.time()

            #update weights
            loss.backward()

            #updated weights
            optimizer.step()

            torch.cuda.synchronize()
            end_bwd = time.time()
            back_pass_time.append(end_bwd - start_fwd)

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    torch.cuda.synchronize()
    end = time.time()
    print("Time to train model: ", end - start)
    print("Average time for forward pass: ", sum(fwd_pass_time)/len(fwd_pass_time))
    print("Average time for backward pass: ", sum(back_pass_time)/len(back_pass_time))
    print("epoch and accuracy during training: ")
    print(epochs)
    print(acc)
    print("Final Accuracy: ", accuracy())

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
    parser.add_argument('--model_family', type=str, default="alexnet", help='alexnet or vgg16')
    parser.add_argument('--load_model_cp', type=str, help='Full path to model *.pt checkpoint file', required=True)
    parser.add_argument('--save_model_cp', type=str, help='Full path to desired location for model *.pt checkpoint file', required=True)
    parser.add_argument('--network_class_file', type=str, help='Name of file which defines model class, omit the *.py ending - class must be named: net() (file must be stored this repo)', required=True)
    parser.add_argument('--data', type=str, default="cifar10", help='Choose dataset: cifar10, fruits, imagenet')
    

    
args = parser.parse_args()
main(args)
