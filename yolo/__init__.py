import gc

from .train import train
from .predict import predict

def main(args):
    gc.collect()
    if args.Action == 'train':
        train()
    elif args.Action == 'predict':
        predict()
    gc.collect()

