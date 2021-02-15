import h5py
import numpy as np

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('h5_filepath', help='Specify the .h5 file path')
#     parser.add_argument('npy_savepath', help='Specify the .npy save path')
#     parser.add_argument('savename', help='Specify the name of the final .npy file')
#     return parser.parse_args()

class ChannelExtraction:
    def __init__(self):
        return

    def channel_extraction(self, loadpath, savepath, savename, action, nr_chn=16):
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