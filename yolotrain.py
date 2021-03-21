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

torch.cuda.empty_cache()

# CONSTANTS
mycfgdir = "cfg/yolovtiny.cfg"
dataPath = "save/jp/final"
myreso = 416

# NETWORK
darknet = DarkNet("cfg/yolov3tiny.cfg", myreso)
# print(darknet.module_list)

# OPTIMIZER & HYPERPARAMETERS
optimizer = optim.SGD(filter(lambda p: p.requires_grad, darknet.parameters()), lr=0.0001, 
    momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# IMAGE PREPROCESSING!!!
# transform = transforms.Compose([
#     # transforms.RandomResizedCrop(size=myreso, interpolation=3),
#     transforms.Resize(size=(myreso, myreso), interpolation=3),
#     transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])
transform = None
# ====================================================

# Train and Test data allocation
trainset = MmwaveDataset(data_dir = dataPath, transforms = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, \
    shuffle=True, num_workers=2, collate_fn = collate)
testset = MmwaveDataset(data_dir = dataPath, transforms = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, \
    shuffle=True, num_workers=2, collate_fn = collate)
# ====================================================

start_epoch = 0
start_iteration = 0
# LOAD A CHECKPOINT!!!
# start_epoch, start_iteration, state_dict = load_checkpoint(
#     'checkpoints',
#     int(start_epoch),
#     int(start_iteration)
# )
# darknet.load_state_dict(state_dict)
# ====================================================

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
darknet.to(device) # Put the network on device
print(next(darknet.parameters()).device)


# TRAIN
print(f'[LOG] TRAIN | Training images: {len(trainset)}')
print(f'[LOG] TRAIN | Test images: {len(testset)}')
print(f'[LOG] TRAIN | Starting to train from epoch {start_epoch} iteration {start_iteration}')
for epoch in range(start_epoch, 10):
    darknet.train(True) # training
    losses = []
    start = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # clear the grads from prev passes
        optimizer.zero_grad()

        inputs = inputs.to(device)      # Images
        targets = targets.to(device)    # Labels
        outputs = darknet(inputs, targets, device) # Loss
        # outputs.register_hook(lambda grad: print(grad))
        outputs['total'].backward()     # Gradient calculations
        
        losses.append(outputs['total'].item())
        optimizer.step()

        end = time.time()

        # Latest iteration!
        # print(f'x: {outputs["x"].item():.2f} y: {outputs["y"].item():.2f} ' \
        #         f'w: {outputs["w"].item():.2f} h: {outputs["h"].item():.2f} ' \
        #         # f'cls: {outputs["cls"].item():.2f} ' \
        #         f'conf: {outputs["conf"].item()}')
        # print(f'x: {outputs["x"].item():.2f} y: {outputs["y"].item():.2f} ')

        if (batch_idx % 100) == 0:
            print(f'[LOG] TRAIN | Batch #{batch_idx} \
                Loss: {np.mean(losses)} \
                Time: {end - start}s')
            start = time.time()

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

    # save_checkpoint('checkpoints/', epoch + 1, 0, {
    #     'epoch': epoch + 1,
    #     'iteration': 0,
    #     'state_dict': darknet.state_dict()
    # })

# torch.save(darknet.state_dict(), f'checkpoints/yolommwave.ckpt')