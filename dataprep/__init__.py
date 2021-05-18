import argparse
import sys, gc

from .channel_extraction import chext
from .processing import proc
from .truth import truth

def parse_arg():
    parser = argparse.ArgumentParser(description='Data preprocessing module', add_help=True)

    parser.add_argument('--pathin', type=str, required=True,
        help="Path for the input folder")
    parser.add_argument('--pathout', type=str,
        help="Path for the output folder")
    parser.add_argument('--saveprefix', type=str,
        help="Prefix for the save file")

    parser.add_argument('--chext', action='store_true',
        help="Perform channel extraction")
    parser.add_argument('--proc', action='store_true',
        help="Perform signal processing (FFT and denoising)")
    parser.add_argument('--truth', action='store_true',
        help="Perform ground truth (clustering, tracking) bouding box calculations")

    
    parser.add_argument('--objcount', type=int, default=1,
        help="Number of objects per image (default: 1)")
    parser.add_argument('--reso', type=int, default=416,
        help="Input image resolution (def: 416)")

    parser.add_argument('--v', type=int, default=0, 
        help="Verbose (0 minimal (def), 1 normal, 2 all")
    
    return parser.parse_args(sys.argv[2:])

def main():
    args = parse_arg()

    if args.chext:
        chext(args)
    gc.collect()
    if args.proc:
        proc(args)
    gc.collect()
    if args.truth:
        truth(args)
    gc.collect()
