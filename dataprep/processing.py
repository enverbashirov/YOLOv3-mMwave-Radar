import numpy as np
import scipy as sp
from scipy import signal, io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import shutil
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from time import sleep


if __name__ == '__main__':
    # # # PARAMETERS INIT # # #

    c0 = 1/np.sqrt(4*np.pi*1e-7*8.85e-12)    # speed of light
    f_start = 76e9
    f_stop = 78e9
    Tramp_up = 180e-6
    Tramp_down = 32e-6
    Tp = 250e-6
    T_int = 66.667e-3
    N = 512
    N_frames = 1250
    N_loop = 256
    Tx_power = 100
    kf = 1.1106e13
    BrdFuSca = 4.8828e-5
    fs = 2.8571e6
    fc = (f_start + f_stop)/2

    # # # CONFIGURE SIGNAL PROCESSING # # # 

    # # Range dimension
    NFFT = 2**10                                                             # number of fft points in range dim
    nr_chn = 16                                                              # number of channels
    # fft will be computed using a hannng window to lower border effects
    win_range = np.broadcast_to(np.hanning(N-1), (N_loop, nr_chn, N-1)).T    # integral of the window for normalization
    sca_win = np.sum(win_range[:, 0, 0])

    v_range = np.arange(NFFT)/NFFT*fs*c0/(2*kf)       # vector of range values for each range bin

    r_min = 0                                         # min range considered
    r_max = 10                                         # max range considered

    arg_rmin = np.argmin(np.abs(v_range - r_min))     # index of the min range considered value
    arg_rmax = np.argmin(np.abs(v_range - r_max))     # index of the max range considered value
    vrange_ext = v_range[arg_rmin:arg_rmax+1]         # vector of range values from rmin to rmax

    # # Angle dimension
    NFFT_ant = 64                                                                    # number of fft points in angle dim
    win_ant = np.broadcast_to(np.hanning(nr_chn).reshape(1,-1,1), (vrange_ext.shape[0], nr_chn, N_loop))
    scawin_ant = np.sum(win_ant[0, :, 0])
    vang_deg = np.arcsin(2*np.arange(-NFFT_ant/2, NFFT_ant/2)/NFFT_ant)/np.pi*180     # vector of considered angles [-90, 90]

    ant_idx = np.arange(nr_chn)
    cal_data = io.loadmat('./calibration.mat')['CalData']               # load complex calibration weights for each antenna element 
    cal_data = cal_data[:16]                                               # keep weights for TX1 only
    mcal_data = np.broadcast_to(cal_data, (N-1, cal_data.shape[0], N_loop))
    
    # # Doppler dimension
    NFFT_vel = 256                                                                    # number of fft points in angle dim
    win_vel = np.broadcast_to(np.hanning(N_loop).reshape(1, 1, -1), (vrange_ext.shape[0], NFFT_ant, N_loop))
    scawin_vel = np.sum(win_vel[0, 0, :])                                             # scaling factor to normalize window
    vfreq_vel = np.arange(-NFFT_vel/2, NFFT_vel/2)/NFFT_vel*(1/Tp)                    # vector of considered frequencies in Doppler dim
    v_vel = vfreq_vel*c0/(2*fc)                                                       # transform freqs into velocities
    v_vel = np.delete(v_vel, np.arange(124, 132))                                     # delete velocities close to 0

    # # # PROCESS THE RDA SLICES FOR EACH FRAME # # #
    folder = "jp"
    rawpath = f'save/{folder}/chext'
    savepath = f'save/{folder}/proc'
    savename = "sub"

    # Create the subsequent save folders
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    # sequences = [1, 2, 3, 4, 5, 6]   # this is just as an example, you should put here the ids of the sequences you want to process
    # sequences = range(0, len(os.listdir(rawpath)))   # this is just as an example, you should put here the ids of the sequences you want to process
    for i, fname in enumerate(os.listdir(rawpath)):
        frawname = fname.split('.')[0]
        print(f'{i+1} / {len(os.listdir(rawpath))} {frawname} {fname}')

        Data_orig = np.load(f'{rawpath}/{fname}')
        print('Original data shape: ', Data_orig.shape) 

        parts = [0, 1, 2, 3]
        SIDELOBE_LEVEL = 0.01
        LINTHR_HIGH = -97
        LINTHR_LOW = -107                 
        thr = 8   # or 15 

        for part in parts:                           # split processing in parts for memory, each track is split in 4
            savename = f'{frawname}_sub_{part}'
            print(f'\n{savename}')

            Data = Data_orig[:, :, part*32000:(part+1)*32000]   # each part has 32k blocks (128k/4)
            split_locs = np.arange(Data.shape[2], step=N_loop, dtype=np.int)[1:]
            Data = np.stack(np.split(Data, split_locs, axis=2)[:-1], axis=-1)  # split data into a sequence of radar cubes
            print('Time-split data shape', Data.shape)
            
            nsteps = Data.shape[-1]        # last dim is time
            rda_data = np.zeros((len(vrange_ext), NFFT_ant, NFFT_vel-8, nsteps), dtype=np.float32)

            for j in range(nsteps):        # loop on the timesteps
                print('Timestep: {t}'.format(t=j+1), end='\r')
                RawRadarCube = Data[1:, :, :, j]

                # Range fft: window, calibration and scaling are applied
                range_profile = np.fft.fft(RawRadarCube*win_range*mcal_data, NFFT, axis=0)*BrdFuSca/sca_win
                rp_ext = range_profile[arg_rmin:arg_rmax+1]  # extract only ranges of interest (0 to 10 m)
                # Angle fft
                range_angle = np.fft.fftshift(np.fft.fft(rp_ext*win_ant, NFFT_ant, axis=1)/scawin_ant, axes=1)
                # Doppler fft
                range_angle_doppler = np.fft.fftshift(np.fft.fft(range_angle*win_vel, NFFT_vel, axis=2)/scawin_vel, axes=2)
                # absolute value + 20log10 to compute power
                range_angle_doppler = np.abs(range_angle_doppler)
                range_angle_doppler = 20*np.log10(range_angle_doppler)

                range_angle_doppler = np.delete(range_angle_doppler, np.arange(124, 132), axis=2)   # delete velocities close to 0

                # at this point you have the RDA representation and you can apply further denoising
                rdep_thr = np.linspace(LINTHR_HIGH, LINTHR_LOW, range_angle_doppler.shape[0]).reshape((-1, 1, 1))
                rdep_thr = np.broadcast_to(rdep_thr, (rdep_thr.shape[0], range_angle_doppler.shape[1], range_angle_doppler.shape[2]))

                range_angle_doppler -= rdep_thr
                range_angle_doppler[range_angle_doppler < 0] = 0

                maxs = np.max(range_angle_doppler, axis=1).reshape(range_angle_doppler.shape[0], 1, range_angle_doppler.shape[2])
                threshold = maxs - SIDELOBE_LEVEL
                rdep_thr_a = np.broadcast_to(threshold, (range_angle_doppler.shape[0], range_angle_doppler.shape[1], range_angle_doppler.shape[2]))
                range_angle_doppler[range_angle_doppler < rdep_thr_a] = 0

                rda_data[:, :, :, j] = range_angle_doppler

            np.save(f'{savepath}/{savename}.npy', np.asarray(rda_data))
            print('Saved RDA shape: ', rda_data.shape)
            del Data, rda_data, split_locs
            sleep(3)
        del Data_orig
        sleep(3)