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
mycfgdir = "cfg/yolov3test.cfg"
dataPath = "save/jp/final"
myreso = 416
batch_size = 8
train_split = 0.8
shuffle = True
num_workers = 2
random_seed = 42

# NETWORK
darknet = DarkNet(mycfgdir, myreso)
pytorch_total_params = sum(p.numel() for p in darknet.parameters() if p.requires_grad)
print('# of params: ', pytorch_total_params)
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

# Train and Validation data allocation
trainloader, validloader = getDataLoaders(dataPath, transform, train_split=train_split, batch_size=batch_size, \
    shuffle=shuffle, num_workers=num_workers, collate_fn=collate, random_seed=random_seed)
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
# print(next(darknet.parameters()).device)


# TRAIN
print(f'[LOG] TRAIN | Training set: {len(trainloader.dataset)}')
print(f'[LOG] TRAIN | Validation set: {len(validloader.dataset)}')
print(f'[LOG] TRAIN | Starting to train from epoch {start_epoch} iteration {start_iteration}')
for epoch in range(start_epoch,2):
    darknet.train() # set network to training mode
    losses = []
    start = time.time()

    for batch_idx, (_, inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()   # clear the grads from prev passes
        inputs, targets = inputs.to(device), targets.to(device) # Images, Labels
        outputs = darknet(inputs, targets, device)  # Loss
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
            print(f'[LOG] TRAIN | Batch #{batch_idx}\
                Loss: {np.mean(losses)}\
                Time: {end - start}s')
            start = time.time()

    scheduler.step()

    # VALIDATION
    with torch.no_grad():
        vlosses = []
        for batch_idx, (_, inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)

            voutputs = darknet(inputs, targets)
            vlosses.append(voutputs['total'].item())

        # Validation loss!
        print(f'[LOG] VALID | Epoch #{epoch}    \
            Loss: {np.mean(vlosses)}')
    # ====================================================

    # save_checkpoint('checkpoints/', epoch + 1, 0, {
    #     'epoch': epoch + 1,
    #     'iteration': 0,
    #     'state_dict': darknet.state_dict()
    # })

save_checkpoint('checkpoints/', epoch + 1, 0, {
    'epoch': epoch + 1,
    'iteration': 0,
    'state_dict': darknet.state_dict()
})