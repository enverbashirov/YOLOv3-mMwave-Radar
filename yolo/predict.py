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

def predict(args):
    # CONSTANTS
    pathcfg = f"cfg/{args.cfg}.cfg"
    pathin = f"{args.pathin}"
    pathout = f"{args.pathout}"
    shuffle = True if args.seed != 0 else False
    num_workers = 2

    # NETWORK
    darknet = DarkNet(pathcfg, args.reso, args.obj, args.nms)
    pytorch_total_params = sum(p.numel() for p in darknet.parameters() if p.requires_grad)
    print('# of params: ', pytorch_total_params)
    if args.v > 0:
        print(darknet.module_list)

    # IMAGE PREPROCESSING!!!
    transform = transforms.Compose([
        transforms.Resize(size=(args.reso, args.reso), interpolation=3),
        transforms.ToTensor()
    ])
    # transform = None
    # ====================================================

    # Test data allocation
    _, testloader = getDataLoaders(pathin, transform, train_split=args.datasplit, batch_size=args.bs, \
        shuffle=shuffle, num_workers=num_workers, collate_fn=collate, random_seed=args.seed)
    # ====================================================

    start_epoch = 2
    start_iteration = 0

    # LOAD A CHECKPOINT!!!
    start_epoch, start_iteration = args.ckpt.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        'save/checkpoints',
        int(start_epoch),
        int(start_iteration)
    )
    darknet.load_state_dict(state_dict)
    # ====================================================

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    darknet.to(device) # Put the network on device
    if args.v > 0:
        print(next(darknet.parameters()).device)

    # PREDICT
    print(f'[LOG] PREDICT | Test set: {len(testloader.dataset)}')
    darknet.eval() # set network to evaluation mode
    with torch.no_grad():
        for _, (paths, inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            predictions = darknet(inputs)

            for idx, path in enumerate(paths):
                print(path)
                # name = path.split('/')[-1].split('[')[0]
                
                try:
                    prediction = predictions[predictions[:, 0] == idx]
                    # print(idx, predictions.size(), prediction.size())
                except Exception:
                    prediction = torch.Tensor([])
                    print(f'[ERROR] TEST | No prediction? {prediction}')
                
                # draw_prediction(path, prediction, darknet.reso, names=[''], save_path=f'{pathout}/{name}.png')
                draw_prediction(path, prediction, targets[idx], darknet.reso, \
                    names=[''], save_path=f'{pathout}/{idx}.png')
            return
    # ====================================================
