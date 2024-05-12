from __future__ import print_function
import time
import random
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np


from pose_resnet import *
from loss import MyLoss




model = PoseResNet(Bottleneck, [3, 4, 6, 3])
model.load_state_dict(torch.load("./pretrained/pose_resnet_50_256x192.pth.tar"),strict=False)

custom_layer = model.custom_layer  
for param in custom_layer.parameters():
    param.data.fill_(0.05)

random_image = torch.randn(1, 3, 256, 192)
random_label = torch.randn(1, 17, 3)
criterion = MyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#for epoch in range(100):

#    for inputs, targets in train_loader:

#        optimizer.zero_grad()

        # Forward pass
#        outputs = model(inputs)

        # Compute the loss
#        loss = criterion(outputs, targets)

        # Backward pass
#        loss.backward()

        # Update the parameters
#        optimizer.step()

    # Optionally, print the loss after each epoch
#    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
optimizer.zero_grad()
output = model(random_image)
loss = criterion(output, random_label)
print("loss!!")
print(loss)
loss.backward
optimizer.step()

torch.save(model.state_dict(), 'saved_model.pth')
print("success")