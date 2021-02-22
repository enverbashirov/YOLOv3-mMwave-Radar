import os, shutil, time
from argparse import ArgumentParser

import matplotlib, matplotlib.patches as patches, matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
from sklearn.cluster import DBSCAN

from .channel_extraction import ChannelExtraction
from .util import Cluster, Supporter, polar2cartesian, cartesian2polar, \
    deg2rad_shift, shift_rad2deg, get_box, IOU_score
from .kalman_tracker import KalmanTracker

def imaging(tracker, cluster, data, labels, full_indices):
    flat_data = np.copy(data.ravel())
    full_data = flat_data[full_indices]
    full_data[labels != cluster.label] = 0 
    flat_data[full_indices] = full_data
    flat_data = flat_data.reshape(data.shape)
    
    print(flat_data.shape)
    ra = flat_data.max(2)
    rd = flat_data.max(1)
    plt.subplot(121)
    plt.imshow(rd, aspect='auto')
    plt.subplot(122)
    plt.imshow(ra, aspect='auto', extent=(np.pi, 0.25065, 0.5, 10))

    plt.scatter(tracker.rtheta[1], tracker.rtheta[0], marker='x', c='r')

    plt.colorbar()
    plt.show()
    plt.close()

def plot(path, data_points, ra, noisy_ramap, t_list, action, index, ranges, angles):
    boxes = np.array([kt.box for kt in t_list])

    angles = deg2rad_shift(angles)
    
    # ramap = data_points.mean(2)

    _, ax = plt.subplots(1, 2)
    ax[0].set_title('Point-cloud representation')
    ax[1].set_title('RA map image representation')
    ax[0].scatter(ra[1], ra[0], marker='.')#, c=labels)
    ax[1].imshow(noisy_ramap, aspect='auto')
    ax[0].set_xlabel(r'$\theta$ [rad]')
    ax[0].set_ylabel(r'$R$ [m]')
    ax[0].set_xlim([0.25065, np.pi])
    ax[0].set_ylim([0.5, 10])
    ax[0].grid()
    for i in range(len(boxes)):
        # add real valued bb on point cloud plot
        add_bb(boxes[i], ax[0], t_list[i].id)
        # add pixel-level bb to ra image
        int_box = adjust_bb(boxes[i], ranges, angles)
        add_bb(int_box, ax[1], t_list[i].id)

    if action == 'save':
        plt.savefig(path + f'fig_{index}', format='png', dpi=300)
        plt.close()
    elif action == 'plot':
        plt.title(f'Frame {index}')
        plt.show()
        plt.close()

def plot4train(path, data_points, noisy_ramap, t_list, action, ranges, angles):
    boxes = np.array([kt.box for kt in t_list])

    angles = deg2rad_shift(angles)
    
    fig = plt.figure(figsize=(13,13), dpi=32, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(noisy_ramap, aspect='auto')

    bb = np.zeros((4, 1))
    for i in range(len(boxes)):
        # # add pixel-level bb to ra image
        bb = adjust_bb(boxes[i], ranges, angles)
        # add_bb(bb, ax, t_list[i].id)

    if action == 'save':
        bb = bb.astype(int)
        plt.savefig(f'{path}_[{bb[0][0]},{bb[1][0]},{bb[2][0]},{bb[3][0]}]', format='png', dpi=32)
    elif action == 'plot':
        plt.show()
    plt.close()

def add_bb(bb, ax, note):
    ax.add_patch(patches.Rectangle((bb[1] - bb[3]/2, bb[0] - bb[2]/2),     # top left corner coordinates
                        bb[3],       # width
                        bb[2],       # height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'))

def adjust_bb(bb_real, r, a):
    '''
    this function is needed to map the bb obtained in real values to the image 
    pixel coordinates without the bias introduced by non-uniform spacing of angle bins
    '''
    bb_ind = np.zeros(bb_real.shape[0])
    bb_ind[0] = np.argmin(np.abs(r - bb_real[0]))
    bb_ind[1] = np.argmin(np.abs(a - bb_real[1]))
    top = np.argmin(np.abs(r - (bb_real[0] - bb_real[2]/2)))
    bottom = np.argmin(np.abs(r - (bb_real[0] + bb_real[2]/2)))
    left = np.argmin(np.abs(a - (bb_real[1] + bb_real[3]/2)))
    right = np.argmin(np.abs(a - (bb_real[1] - bb_real[3]/2)))
    bb_ind[2] = np.abs(top - bottom)
    bb_ind[3] = np.abs(left - right)
    return bb_ind.reshape(-1, 1)