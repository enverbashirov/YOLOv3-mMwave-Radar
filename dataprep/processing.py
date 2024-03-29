import os, shutil, gc
from argparse import ArgumentParser
from time import sleep

import h5py
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import io, signal
from scipy.signal.windows import nuttall, taylor

from .util import *

def proc(args):
    rawpath = f'dataset/{args.pathin}/chext'
    savepath = f'dataset/{args.pathout}/proc' if args.pathout else f'dataset/{args.pathin}/proc'
    print(f'[LOG] Proc | Starting: {args.pathin}')

    # Create the subsequent save folders
    # if os.path.isdir(savepath):
    #     shutil.rmtree(savepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath + '/raw/')
        os.mkdir(savepath + '/denoised/')

    # # # PARAMETERS INIT # # #

    c0 = 1/np.sqrt(4*np.pi*1e-7*8.85e-12)    # speed of light
    f_start = 76e9
    f_stop = 78e9
    # Tramp_up = 180e-6
    # Tramp_down = 32e-6
    Tp = 250e-6
    # T_int = 66.667e-3
    N = 512
    # N_frames = 1250
    N_loop = 256
    # Tx_power = 100
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
    # print(win_range.shape)
    sca_win = np.sum(win_range[:, 0, 0])

    v_range = np.arange(NFFT)/NFFT*fs*c0/(2*kf)       # vector of range values for each range bin

    r_min = 0                                         # min range considered
    r_max = 10                                         # max range considered

    arg_rmin = np.argmin(np.abs(v_range - r_min))     # index of the min range considered value
    arg_rmax = np.argmin(np.abs(v_range - r_max))     # index of the max range considered value
    vrange_ext = v_range[arg_rmin:arg_rmax+1]         # vector of range values from rmin to rmax

    # # Doppler dimension
    NFFT_vel = 256                                                                    # number of fft points in angle dim
    win_vel = np.broadcast_to(np.hanning(N_loop).reshape(1, 1, -1), (vrange_ext.shape[0], nr_chn, N_loop))
    scawin_vel = np.sum(win_vel[0, 0, :])
    vfreq_vel = np.arange(-NFFT_vel/2, NFFT_vel/2)/NFFT_vel*(1/Tp)                    # vector of considered frequencies in Doppler dim
    v_vel = vfreq_vel*c0/(2*fc)                                                        # transform freqs into velocities
    v_vel = np.delete(v_vel, np.arange(124, 132))                                     # delete velocities close to 0

    # # Angle dimension
    NFFT_ant = 64                                                                    # number of fft points in angle dim
    win_ant = np.broadcast_to(taylor(nr_chn, nbar=20, sll=20).reshape(1,-1,1), (vrange_ext.shape[0], nr_chn, NFFT_vel))
    scawin_ant = np.sum(win_ant[0, :, 0])
    # win_ant = np.tile(win_ant, (len(vrange_ext), 1))
    # vang_deg = np.arcsin(2*np.arange(-NFFT_ant/2, NFFT_ant/2)/NFFT_ant)/np.pi*180     # vector of considered angles [-90, 90-dtheta]
    # print(vang_deg)
    # print(deg2rad_shift(vang_deg))
    
    # ant_idx = np.concatenate([np.arange(nr_chn), np.arange(nr_chn+1, 2*nr_chn)])      # indices of virtual antenna elements
    # ant_idx = np.arange(nr_chn)
    cal_data = io.loadmat('dataprep/calibration.mat')['CalData']               # load complex calibration weights for each antenna element 
    cal_data = cal_data[:16]                                                           # keep weights for TX1 only
    mcal_data = np.broadcast_to(cal_data, (N-1, cal_data.shape[0], N_loop))
    
    # # # PROCESS THE RDA SLICES FOR EACH FRAME # # #
    # sequences = [1, 2, 3, 4, 5, 6]   # this is just as an example, you should put here the ids of the sequences you want to process
    # sequences = range(0, len(os.listdir(rawpath)))   # this is just as an example, you should put here the ids of the sequences you want to process
    for i, fname in enumerate(os.listdir(rawpath)):
        frawname = fname.split('.')[0]
        logprefix = f'[LOG] Proc | {i+1} / {len(os.listdir(rawpath))} {frawname}'
        print(f'{logprefix} {fname}', end='\r')

        Data_orig = np.load(f'{rawpath}/{fname}')
        # print(f'{logprefix} Original data shape: {Data_orig.shape}', end='\r') 

        parts = [0, 1, 2, 3]
        SIDELOBE_LEVEL = 3
        LINTHR_HIGH = -97
        LINTHR_LOW = -107                 

        for part in parts:                           # split processing in parts for memory, each track is split in 4
            savename = f'{args.saveprefix}_seq_{frawname.split("_")[2]}_sub_{part}' \
                if args.saveprefix else f'{frawname}_sub_{part}'
            logprefix = f'[LOG] Proc | {i*len(parts)+part+1} / {len(os.listdir(rawpath))*len(parts)} {frawname}'
            print(f'{logprefix} {savename}', end='\r')

            Data = Data_orig[:, :, part*32000:(part+1)*32000]   # each part has 32k blocks (128k/4)
            split_locs = np.arange(Data.shape[2], step=N_loop, dtype=np.int)[1:]
            Data = np.stack(np.split(Data, split_locs, axis=2)[:-1], axis=-1)  # split data into a sequence of radar cubes
            print(f'{logprefix} Time-split \t\t\t', end='\r')
            
            nsteps = Data.shape[-1]        # last dim is time
            rda_data = np.zeros((len(vrange_ext), NFFT_ant, NFFT_vel, nsteps), dtype=np.float32)
            raw_ra = np.zeros((len(vrange_ext), NFFT_ant, nsteps), dtype=np.float32)
            for j in range(nsteps):        # loop on the timesteps
                print(f'{logprefix} Timestep: {j+1} \t\t\t', end='\r')
                RawRadarCube = Data[1:, :, :, j]
                # print(RawRadarCube.shape)
                # Range fft: window, calibration and scaling are applied
                range_profile = np.fft.fft(RawRadarCube*win_range*mcal_data, NFFT, axis=0)*BrdFuSca/sca_win
                rp_ext = range_profile[arg_rmin:arg_rmax+1]  # extract only ranges of interest (0 to 10 m)
                # background subtraction for MTI
                rp_ext -= np.mean(rp_ext, axis=2, keepdims=True)
                # Doppler fft
                range_doppler = np.fft.fftshift(np.fft.fft(rp_ext*win_vel, NFFT_vel, axis=2)/scawin_vel, axes=2)
                # Angle fft
                range_angle_doppler = np.fft.fftshift(np.fft.fft(range_doppler*win_ant, NFFT_ant, axis=1)/scawin_ant, axes=1)
                
                # absolute value + 20log10 to compute power
                range_angle_doppler = 20*np.log10(np.abs(range_angle_doppler))

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(range_angle_doppler.max(2))
                # ax[1].imshow(range_angle_doppler.max(1))
                # plt.show()

                raw_ra[..., j] = range_angle_doppler.max(2)  # store raw range-angle image

                # at this point you have the RDA representation and you can apply further denoising
                rdep_thr = np.linspace(LINTHR_HIGH, LINTHR_LOW, range_angle_doppler.shape[0]).reshape((-1, 1, 1))
                
                range_angle_doppler -= rdep_thr
                range_angle_doppler[range_angle_doppler < 0] = 0

                maxs = np.max(range_angle_doppler, axis=1).reshape(range_angle_doppler.shape[0], 1, range_angle_doppler.shape[2])
                # maxs = np.max(range_angle_doppler, axis=(0, 2)).reshape(1, range_angle_doppler.shape[1], 1)
                threshold = maxs - SIDELOBE_LEVEL
                range_angle_doppler[range_angle_doppler < threshold] = 0

                rda_data[..., j] = range_angle_doppler

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(range_angle_doppler.max(2))
                # ax[1].imshow(range_angle_doppler.max(1))
                # plt.show()

            print(f'{logprefix} Saving: {savename} \t\t\t')
            np.save(f'{savepath}/denoised/{savename}.npy', rda_data)
            np.save(f'{savepath}/raw/{savename}.npy', raw_ra)

            del Data, rda_data, split_locs, raw_ra
            gc.collect()
        del Data_orig
        gc.collect()
    print('\n')