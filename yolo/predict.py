import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
import torchvision.transforms as transforms

import os, sys 
# import pickle, time, random

import numpy as np
# from PIL import Image
import argparse

from torchvision.transforms.functional import InterpolationMode

from .darknet import DarkNet
from .dataset import *
from .util import *

def parse_arg():
    parser = argparse.ArgumentParser(description='MmWaveYoLo Prediction module', add_help=True)

    parser.add_argument('--cfg', type=str, default='yolov3micro',
        help="Name of the network config (default: yolov3micro)")
    parser.add_argument('--pathin', type=str,
        help="Path for the input folder (default: testset)")
    parser.add_argument('--pathout', type=str,
        help="Path for the output folder")
    parser.add_argument('--video', type=str, default='False',
        help="Create video after prediction (default: False)")
        
    parser.add_argument('--datasplit', type=float, default=0, 
        help="Dataset split percentage (default: 0 (single set))")
    parser.add_argument('--seq', type=int, default=1, 
        help="Number of images per sequence (default: 1)")
    parser.add_argument('--seed', type=float, default=0, 
        help="Seed for the random shuffling (default: 0, (no shuffle))")
    parser.add_argument('--bs', type=int, default=8, 
        help="Batch size (default: 8)")
    parser.add_argument('--ckpt', type=str, default='10.0',
        help="Checkpoint name <'epoch'.'iteration'>")

    parser.add_argument('--nms', type=float, default=0.5, 
        help="NMS threshold (default: 0.5)")
    parser.add_argument('--obj', type=float, default=0.5, 
        help="Objectiveness threshold (default: 0.5)")
    parser.add_argument('--iou', type=float, default=0.5, 
        help="Intersection over Union threshold (default: 0.5)")
    parser.add_argument('--reso', type=int, default=416,
        help="Input image resolution (default: 416)")

    parser.add_argument('--v', type=int, default=0, 
        help="Verbose (0 minimal (default), 1 normal, 2 all")
    
    return parser.parse_args(sys.argv[2:])

def predict():
    torch.cuda.empty_cache()
    
    # CONSTANTS
    args = parse_arg()
    pathcfg = f"cfg/{args.cfg}.cfg"
    pathin = f"dataset/{args.pathin}/final"
    pathout = f"results/{args.pathout}"
    num_workers = 2

    # NETWORK
    darknet = DarkNet(pathcfg, args.reso, args.obj, args.nms)
    pytorch_total_params = sum(p.numel() for p in darknet.parameters() if p.requires_grad)
    print('# of params: ', pytorch_total_params)
    if args.v > 1:
        print(darknet.module_list)

    # IMAGE PREPROCESSING!!!
    transform = False
    # ====================================================

    # Test data allocation
    _, testloader = getDataLoaders(pathin, transform, reso=args.reso, \
        train_split=args.datasplit, \
        batch_size=args.bs, seq=args.seq, num_workers=num_workers, \
        collate_fn=collate, random_seed=args.seed)
    # ====================================================

    start_epoch = 2
    start_iteration = 0

    # LOAD A CHECKPOINT!!!
    start_epoch, start_iteration = args.ckpt.split('.')
    start_epoch, start_iteration, state_dict, _, _, _, _ = load_checkpoint(
        f'save/checkpoints_test/',
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

    # Create the subsequent save folders
    # if os.path.isdir(pathout):
    #     shutil.rmtree(pathout)
    if not os.path.isdir(pathout):
        os.makedirs(pathout)

    # PREDICT
    print(f'[LOG] PREDICT | Test set: {len(testloader.dataset)}')
    darknet.eval() # set network to evaluation mode
    outcomes = np.zeros(4)
    predList = []
    countLabels = 0
    predcount = 0

    with torch.no_grad():
        for bidx, (paths, inputs, targets) in enumerate(testloader):
            paths = np.array(paths).T
            # print(bidx, np.shape(paths), inputs.shape, targets.shape)

            inputs = inputs.to(device)
            predictions = darknet(inputs)

            for idx, path in enumerate(paths):
                print(f'[LOG] PREDICT | Predicting {(bidx*args.bs)+idx+1}/{len(testloader.dataset)}', end='\r')
                savename = path[-1].split('/')[-1].split('_')[2]
                
                try:
                    prediction = predictions[predictions[:, 0] == idx]
                except Exception:
                    prediction = torch.Tensor([])
                    if args.v > 0:
                        print(f'[ERROR] TEST | No prediction? {prediction}')

                predL, outcome = correctness(prediction, targets[idx,-1,...], reso=darknet.reso, iou_thresh=args.iou)
                predList.extend(predL)

                outcomes += outcome
                predcount += prediction.shape[0]
                countLabels += np.count_nonzero(np.count_nonzero(targets[idx,-1,...].numpy(), axis=1))

                draw_prediction(path[-1], prediction, targets[idx,-1,...], darknet.reso, \
                    names=[''], pathout=f'{pathout}/preds', savename=f'{savename}.png')

    if args.video:
        animate_predictions(pathout, args.video)

    # print(len(predList))
    # print(countLabels)
    exit()
    predArr = precision_recall(predList, countLabels)
    metrics, interArr = evaluation_metrics(predArr, countLabels, outcomes)
    plot_precision_recall(predArr, np.asarray(interArr), metrics, pathout=f'{pathout}/map', savename=f'precRecallIoU{args.iou}Obj{args.obj}.png')
    # ====================================================
