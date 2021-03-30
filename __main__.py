import argparse

import yolo
import dataprep

def parse_arg():
    parser = argparse.ArgumentParser(description='mmWave YOLOv3', add_help=True)
    parser.add_argument('Action', type=str, 
        help='Input action (dataprep, train, predict)')

    parser.add_argument('--cfg', type=str, default='yolov3micro',
        help="Name of the network config")
    parser.add_argument('--pathin', type=str, default='save/jp/final',
        help="Path for the input folder")
    parser.add_argument('--pathout', type=str, default='save/results',
        help="Path for the output folder")

    parser.add_argument('--datasplit', type=float, default=0.8, 
        help="Dataset split percentage (def: 0.8 (80 train, 20 val))")
    parser.add_argument('--seed', type=float, default=42, 
        help="Seed for the random shuffling (def: 42)")
    parser.add_argument('--bs', type=int, default=8, 
        help="Batch size")
    parser.add_argument('--ckpt', type=str, default='-1.-1',
        help="Checkpoint name (epoch.iteration)")
    parser.add_argument('--ep', type=int, default=5,
        help="Total epoch number (def: 5)")

    parser.add_argument('--lr', type=float, default=1e-4, 
        help="Learning rate (def: 0.0001)")
    parser.add_argument('--nms', type=float, default=0.5, 
        help="NMS threshold (def: 0.5)")
    parser.add_argument('--obj', type=float, default=0.5, 
        help="Objectiveness threshold (def: 0.5)")
    parser.add_argument('--reso', type=int, default=416,
        help="Input image resolution (def: 416)")

    parser.add_argument('--v', type=int, default=0, 
        help="Verbose (0 minimal (def), 1 normal, 2 all")
    
    return parser.parse_args()


args = parse_arg()

if args.Action == 'train' or args.Action == 'predict':
    yolo.main(args)
elif args.Action == 'data':
    dataprep.main(args)
