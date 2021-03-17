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
dataPath = "save/jp/final"
myreso = 416

# NETWORK
darknet = DarkNet("cfg/yolov3tiny.cfg", myreso)
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
darknet.to(device) # Put the network on device

# OPTIMIZER & HYPERPARAMETERS
optimizer = optim.SGD(filter(lambda p: p.requires_grad, darknet.parameters()), lr=0.001, 
    momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# DATA PREPARATION
# transform = transforms.Compose([
#     # transforms.RandomResizedCrop(size=myreso, interpolation=3),
#     transforms.Resize(size=(myreso, myreso), interpolation=3),
#     transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])
transform = None
# Train and Test data allocation
trainset = MmwaveDataset(data_dir = dataPath, transforms = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, \
    shuffle=True, num_workers=2, collate_fn = collate)
testset = MmwaveDataset(data_dir = dataPath, transforms = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, \
    shuffle=True, num_workers=2, collate_fn = collate)

# TRAIN
for epoch in range(10):  # loop over the dataset multiple times
    darknet.train(True) # training
    # darknet.train(False) # detection

    losses = []

    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # clear the grads from prev passes
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)
        print(targets)

        outputs = darknet(inputs, targets, device)
        outputs['total'].backward()
        losses.append(outputs['total'].item())

        # if type(outputs) is dict:

        optimizer.step()

        end = time.time()
        print(f'x: {outputs["x"].item():.2f} y: {outputs["y"].item():.2f} ' \
                f'w: {outputs["w"].item():.2f} h: {outputs["h"].item():.2f} ' \
                f'cls: {outputs["cls"].item():.2f} conf: {outputs["conf"].item():.2f}')

        if batch_idx % 100 == 0:
            # Loss : {np.mean(losses)} \
            print(f'Batch Index : {batch_idx} \
                Time : {end - start} seconds')

            start = time.time()

        # darknet.eval()
        # total = 0
        # correct = 0
    scheduler.step()


    # with torch.no_grad():
    #   for batch_idx, (inputs, targets) in enumerate(testloader):
    #       inputs, targets = inputs.to(device), targets.to(device)

    #       s = darknet(inputs, device)
    #       _, predicted = torch.max(outputs.data, 1)
    #       total += targets.size(0)
    #       correct += predicted.eq(targets.data).cpu().sum()

    #   print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct/total))
    #   print('--------------------------------------------------------------')
    # darknet.train()  
        
torch.save(darknet.state_dict(), 'checkpoints/yolommwave_net.pth')