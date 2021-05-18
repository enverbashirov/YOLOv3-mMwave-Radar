import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle, os, time, random, sys
from PIL import Image
import argparse

from .darknet import DarkNet
from .dataset import *
from .util import *

def parse_arg():
    parser = argparse.ArgumentParser(description='mmWaveYoLov3 Training module', add_help=True)

    parser.add_argument('--cfg', type=str, default='yolov3micro',
        help="Name of the network config")
    parser.add_argument('--pathin', type=str, default='trainset',
        help="Input dataset name")

    parser.add_argument('--datasplit', type=float, default=0.8, 
        help="Dataset split percentage (def: 0.8 (80 (train):20 (validation))")
    parser.add_argument('--seed', type=float, default=42, 
        help="Seed for the random shuffle (default: 42, 0 for no shuffling)")
    parser.add_argument('--bs', type=int, default=8, 
        help="Batch size (default: 8, 0 for single batch)")
    parser.add_argument('--ckpt', type=str, default='0.0',
        help="Checkpoint name as <'epoch'.'iteration'>")
    parser.add_argument('--ep', type=int, default=5,
        help="Total epoch number (default: 5)")

    parser.add_argument('--lr', type=float, default=1e-5, 
        help="Learning rate (default: 1e-5)")
    parser.add_argument('--reso', type=int, default=416,
        help="Input image resolution (default: 416)")

    parser.add_argument('--v', type=int, default=0, 
        help="Verbose (0 minimal (default), 1 normal, 2 all")
    
    return parser.parse_args(sys.argv[2:])

def train():
    torch.cuda.empty_cache()

    # CONSTANTS
    args = parse_arg()
    pathcfg = f"cfg/{args.cfg}.cfg"
    pathin = f"dataset/{args.pathin}/final"
    num_workers = 2

    # NETWORK
    darknet = DarkNet(pathcfg, args.reso)
    pytorch_total_params = sum(p.numel() for p in darknet.parameters() if p.requires_grad)
    print('# of params: ', pytorch_total_params)
    if args.v > 0:
        print(darknet.module_list)

    # LOAD A CHECKPOINT!!!
    start_epoch, start_iteration = [0, 0]
    tlosses, vlosses = [], []
    optimizer, scheduler = None, None
    start_epoch, start_iteration = [int(x) for x in args.ckpt.split('.')]
    if start_epoch != 0 and start_epoch != 0:
        start_epoch, start_iteration, state_dict, \
        tlosses, vlosses, \
        optimizer, scheduler = load_checkpoint(
            f'save/checkpoints/',
            int(start_epoch),
            int(start_iteration)
        )
        darknet.load_state_dict(state_dict)
    # ====================================================

    # OPTIMIZER & HYPERPARAMETERS
    if optimizer == None:
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, darknet.parameters()), \
        #     lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, darknet.parameters()), \
            lr=args.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=0, amsgrad=False)
    if scheduler == None:
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
        num_workers=num_workers, collate_fn=collate, random_seed=args.seed)
    # ====================================================

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1: # Use Multi GPU if available
        darknet = nn.DataParallel(darknet)
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
            save_checkpoint(f'save/checkpoints/', epoch+1, 0, {
                'epoch': epoch+1,
                'iteration': 0,
                'state_dict': darknet.state_dict(),
                'tlosses': tlosses,
                'vlosses': vlosses,
                'optimizer': optimizer,
                'scheduler': scheduler
            })
            plot_losses(tlosses, vlosses, f'save/losses')

    save_checkpoint(f'save/checkpoints/', epoch+1, 0, {
        'epoch': epoch+1,
        'iteration': 0,
        'state_dict': darknet.state_dict(),
        'tlosses': tlosses,
        'vlosses': vlosses,
        'optimizer': optimizer,
        'scheduler': scheduler
    })
    plot_losses(tlosses, vlosses, f'save/losses')

