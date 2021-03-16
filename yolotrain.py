import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle, os, time, random
from PIL import Image

from yolo import *

# CONSTANTS
mycfgdir = "cfg/yolovtiny.cfg"
myreso = 416

dataPath = "save/jp/final"

# If running on Windows and you get a BrokenPipeError, try setting
# the num_worker of torch.utils.data.DataLoader() to 0.
trainset = MmwaveDataset(data_dir = dataPath, transforms = None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
testset = MmwaveDataset(data_dir = dataPath, transforms=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

# Define the network
net = DarkNet("cfg/yolov3tiny.cfg", myreso)
net.train(True) # training
# net.train(False) # detection

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device) # Put the network on device

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# hyperparameters
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

# TRAIN
for epoch in range(10):  # loop over the dataset multiple times
    losses = []
    
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # print(inputs.size())
        # print(targets.size())
        
        outputs = net(inputs, targets, device)
        print(outputs.size())
        exit()

        # compute gradients for all output params 
        # (could be modified for only specific params)
        outputs.requires_grad = True

        # clear the grads from prev passes
        optimizer.zero_grad()

        # compute dloss/dx for every requires_grad=True
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        end = time.time()

        if batch_idx % 100 == 0:
            print(f'Batch Index : {batch_idx} \
                Loss : {np.mean(losses)} \
                Time : {end - start} seconds')

            start = time.time()

        net.eval()
        total = 0
        correct = 0

    scheduler.step()

    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          inputs, targets = inputs.to(device), targets.to(device)

          s = net(inputs, device)
          _, predicted = torch.max(outputs.data, 1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum()

      print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct/total))
      print('--------------------------------------------------------------')
    net.train()  
        
torch.save(net.state_dict(), 'test/yolommwave_net.pth')