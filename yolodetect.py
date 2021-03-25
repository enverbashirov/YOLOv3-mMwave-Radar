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
mycfgdir = "cfg/test.cfg"
dataPath = "save/jp/final"
savePath = "test/results"
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
#     transforms.Resize(size=(myreso, myreso), interpolation=3),
#     transforms.ToTensor()
# ])
transform = None
# ====================================================

# Test data allocation
_, testloader = getDataLoaders(dataPath, transform, train_split=train_split, batch_size=batch_size, \
    shuffle=shuffle, num_workers=num_workers, collate_fn=collate, random_seed=random_seed)
# ====================================================

start_epoch = 2
start_iteration = 0
# LOAD A CHECKPOINT!!!
start_epoch, start_iteration, state_dict = load_checkpoint(
    'checkpoints',
    int(start_epoch),
    int(start_iteration)
)
darknet.load_state_dict(state_dict)
# ====================================================

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
darknet.to(device) # Put the network on device
# print(next(darknet.parameters()).device)

# TEST
print(f'[LOG] TEST | TEST set: {len(testloader.dataset)}')
darknet.eval() # set network to evaluation mode
with torch.no_grad():
    for batch_idx, (paths, inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        predictions = darknet(inputs)

        for idx, path in enumerate(paths):
            print(path)
            name = path.split('/')[-1].split('[')[0]
            
            try:
                prediction = predictions[predictions[:, 0] == 0]
                # print(prediction.size())
            except Exception:
                prediction = torch.Tensor([])
                print(f'[ERROR] TEST | No prediction? {prediction}')
            
            # draw_detection(path, prediction, darknet.reso, names=[''], save_path=f'{savePath}/{name}.png')
            draw_detection(path, prediction, darknet.reso, names=[''], save_path=f'{savePath}/{idx}.png')
            exit()


# ====================================================
