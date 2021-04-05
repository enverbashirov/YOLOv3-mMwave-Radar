from .train import train
from .predict import predict

def main(args):
    if args.Action == 'train':
        train()
    elif args.Action == 'predict':
        predict()

