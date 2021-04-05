import h5py
import numpy as np
import os, shutil

def chext(args):
    rawpath = f'raw/{args.pathin}'
    savepath = f'save/{args.pathout}/chext' if args.pathout else f'save/{args.pathin}/chext'
    print(f'[LOG] ChExt | Starting: {args.pathin}')

    # Create the subsequent save folders
    # if os.path.isdir(savepath):
    #     shutil.rmtree(savepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    for i, fname in enumerate(os.listdir(rawpath)):
        logprefix = f'[LOG] ChExt | {i+1} / {len(os.listdir(rawpath))}'
        print(f'{logprefix} fname', end='\r')
        channel_extraction(
            f'{rawpath}/{fname}',
            savepath,
            f'{args.saveprefix}_seq_{i}' if args.saveprefix else f'{fname.split("_")[0]}_seq_{fname.split("_")[1].split(".")[0]}',
            action='SAVE',
            logprefix=logprefix)
    print('\n')

def channel_extraction(loadpath, savepath, savename, action, logprefix='', nr_chn=16):
    with h5py.File(loadpath, 'r+') as h5data:
        print(f'{logprefix} Initializing: {loadpath}', end='\r')
        Data = np.zeros((h5data['Chn1'].shape[1], nr_chn, h5data['Chn1'].shape[0]), dtype=np.float32)
        for i in range(nr_chn):
            print(f'{logprefix} Extracting channel {i+1} \t\t\t', end='\r')
            channel = np.asarray(h5data['Chn{}'.format(i+1)])
            Data[:, i, :] = channel.T
        print(f'{logprefix} Finalizing {savepath}', end='\r')
        if action == 'SAVE':
            print(f'{logprefix} Saving', end='\r')
            np.save(f'{savepath}/{savename}', Data)
            print(f'{logprefix} Saved: {savepath}/{savename} Data shape: {Data.shape}')
        elif action == 'RETURN':
            return Data
        else:
            print(f'[ERR] ChExt | Invalid action, please select SAVE or RETURN')