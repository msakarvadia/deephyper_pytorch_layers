import torch
import torchvision
from torch import nn


#Now using the AlexNet
class net(nn.Module):
#TODO Add This

print("Creating model")
AlexNet_model = net()

print("Saving model")
# Additional information
PATH = "/home/mansisak/deephyper_pytorch_layers/alexnet/alexnet_models/performant_models/imagenet/alexnet_performant_emodel_0_epoch.pt"
torch.save({'model_state_dict': AlexNet_model.state_dict()}, PATH)

print("Model saved")
