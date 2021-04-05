import argparse
import sys

import yolo
import dataprep

def parse_arg():
    parser = argparse.ArgumentParser(description='mmWave YOLOv3', add_help=True,
        usage='''python . <action> [<args>]

        Actions:
            train       Network training module
            predict     Object detection module
            dataprep    Data preprocessing module
        '''
        )
    parser.add_argument('Action', type=str, help='Action to run')

    return parser.parse_args(sys.argv[1:2])

args = parse_arg()

if args.Action == 'train' or args.Action == 'predict':
    yolo.main(args)
elif args.Action == 'dataprep':
    dataprep.main()
else:
    print('Unknown action. Check "python . --help"')
