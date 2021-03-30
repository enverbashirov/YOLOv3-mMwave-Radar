import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle, os, time, random
from PIL import Image

from .darknet import DarkNet
from .dataset import *
from .util import *

torch.cuda.empty_cache()

def train(args):
    # CONSTANTS
    pathcfg = f"cfg/{args.cfg}.cfg"
    pathin = f"{args.pathin}"
    # pathout = f"{args.pathout}"
    shuffle = True if args.seed != 0 else False
    num_workers = 2

    # NETWORK
    darknet = DarkNet(pathcfg, args.reso, args.obj, args.nms)
    pytorch_total_params = sum(p.numel() for p in darknet.parameters() if p.requires_grad)
    print('# of params: ', pytorch_total_params)
    if args.v > 0:
        print(darknet.module_list)

    # OPTIMIZER & HYPERPARAMETERS
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, darknet.parameters()), \
        lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # IMAGE PREPROCESSING!!!
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=args.reso, interpolation=3),
        transforms.Resize(size=(args.reso, args.reso), interpolation=3),
        transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    # ====================================================

    # Train and Validation data allocation
    trainloader, validloader = getDataLoaders(pathin, transform, \
        train_split=args.datasplit, batch_size=args.bs, \
        shuffle=shuffle, num_workers=num_workers, \
        collate_fn=collate, random_seed=args.seed)
    # ====================================================

    # LOAD A CHECKPOINT!!!
    start_epoch, start_iteration = args.ckpt.split('.')
    if start_epoch != '-1' and start_epoch != '0':
        start_epoch, start_iteration, state_dict = load_checkpoint(
            'save/checkpoints',
            int(start_epoch),
            int(start_iteration)
        )
        darknet.load_state_dict(state_dict)
    else:
        start_epoch, start_iteration = [0, 0]
    # ====================================================

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    darknet.to(device) # Put the network on device
    if args.v > 0:
        print(next(darknet.parameters()).device)

    # TRAIN
    print(f'[LOG] TRAIN | Training set: {len(trainloader.dataset)}')
    print(f'[LOG] TRAIN | Validation set: {len(validloader.dataset)}')
    print(f'[LOG] TRAIN | Starting to train from epoch {start_epoch} iteration {start_iteration}')
    if start_epoch > args.ep:
        print(f'[ERR] TRAIN | Total epochs ({args.ep}) is less then current epoch ({start_epoch})')
        return  

    tlosses, vlosses = [], []
    for epoch in range(start_epoch, args.ep):
        print(f'[LOG] TRAIN | Starting Epoch #{epoch+1}')
        darknet.train() # set network to training mode
        tloss, vloss = [], []
        start = time.time()

        for batch_idx, (_, inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()   # clear the grads from prev passes
            inputs, targets = inputs.to(device), targets.to(device) # Images, Labels
            outputs = darknet(inputs, targets, device)  # Loss
            outputs['total'].backward()     # Gradient calculations
            
            tloss.append(outputs['total'].item())
            optimizer.step()

            end = time.time()

            # print(f'conf: {outputs["conf"].item():.2f}')
            # Latest iteration!
            if args.v == 1:
                print(f'x: {outputs["x"].item():.2f} y: {outputs["y"].item():.2f} ')
            elif args.v == 2:
                print(f'x: {outputs["x"].item():.2f} y: {outputs["y"].item():.2f} ' \
                        f'w: {outputs["w"].item():.2f} h: {outputs["h"].item():.2f} ' \
                        f'cls: {outputs["cls"].item():.2f} ' \
                        f'conf: {outputs["conf"].item()}')

            if (batch_idx % 100) == 99:
                print(f'[LOG] TRAIN | Batch #{batch_idx+1}\
                    Loss: {np.mean(tloss)}\
                    Time: {end - start}s')
                start = time.time()

            # return
        # Save train loss for the epoch
        tlosses.append(np.mean(tloss))

        scheduler.step()

        # VALIDATION
        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)

                voutputs = darknet(inputs, targets)
                vloss.append(voutputs['total'].item())

            # Validation loss!
            print(f'[LOG] VALID | Epoch #{epoch+1}    \
                Loss: {np.mean(vloss)}')
        # Save valid loss for the epoch
        vlosses.append(np.mean(vloss))
        # ====================================================

        if (epoch % 10) == 9:
            save_checkpoint('save/checkpoints/', epoch+1, 0, {
                'epoch': epoch+1,
                'iteration': 0,
                'state_dict': darknet.state_dict()
            })
            plot_losses(tlosses, vlosses)

    save_checkpoint('save/checkpoints/', epoch+1, 0, {
        'epoch': epoch+1,
        'iteration': 0,
        'state_dict': darknet.state_dict()
    })
    plot_losses(tlosses, vlosses)

