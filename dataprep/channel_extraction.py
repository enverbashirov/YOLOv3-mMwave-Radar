import numpy as np
import os
import sys
import h5py
from argparse import ArgumentParser
import shutil

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('h5_filepath', help='Specify the .h5 file path')
    parser.add_argument('npy_savepath', help='Specify the .npy save path')
    parser.add_argument('savename', help='Specify the name of the final .npy file')
    return parser.parse_args()

def channel_extraction(loadpath, savepath, savename, action, nr_chn=16):
    with h5py.File(loadpath, 'r+') as h5data:
        Data = np.zeros((h5data['Chn1'].shape[1], nr_chn, h5data['Chn1'].shape[0]), dtype=np.float32)
        for i in range(nr_chn):
            print('Extracting channel {}'.format(i+1), end='\r')
            channel = np.asarray(h5data['Chn{}'.format(i+1)])
            Data[:, i, :] = channel.T
        if action == 'SAVE':
            print('\nSaving...')
            np.save(f'{savepath}/{savename}', Data)
            print('Saved. Data shape: {}'.format(Data.shape))
        elif action == 'RETURN':
            return Data
        else:
            print('Invalid action, please select SAVE or RETURN')

if __name__ == '__main__':

    # args = parse_args()
    # loadpath = args.h5_filepath
    # savepath = args.npy_savepath
    # savename = args.savename

    folder = "jp"
    rawpath = "raw"
    savepath = f'save/{folder}/chext'
    savename = "seq"

    # Create the subsequent save folders
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    for i, fname in enumerate(os.listdir(f'{rawpath}/{folder}')):
        print(fname)
        channel_extraction(
            f'{rawpath}/{folder}/{fname}', 
            savepath, 
            f'{savename}_{i}', 
            action='SAVE')

