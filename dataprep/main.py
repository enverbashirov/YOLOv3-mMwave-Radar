import os
import shutil
import time
from argparse import ArgumentParser

import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.cluster import DBSCAN

from kalman_tracker import KalmanTracker
from utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path', help='Specify the path to the RDA maps')
    return parser.parse_args()

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
    plt.imshow(ra, aspect='auto')
    plt.colorbar()
    plt.show()
    plt.close()

def plot(path, data_points, t_list, labels, action, index):
    # data_points = np.asarray([polar2cartesian(x) for x in data_points.T])
    # data_points = np.expand_dims(data_points.T, -1)
    # centers = np.array([kt.xy for kt in t_list])
    centers = np.array([kt.rtheta for kt in t_list])
    boxes = np.array([kt.box for kt in t_list])

    fig, ax = plt.subplots()
    # ax.scatter(data_points[:, 1, 0], data_points[:, 0, 0], marker='.', c=labels)
    ax.imshow(np.flipud(data_points.mean(2)), extent=(np.pi, 0, 0.5, 10))
    # ax.xlabel(r'$x$ [m]')
    # ax.ylabel(r'$y$ [m]')
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$R$ [m]')
    # ax.grid()
    # ax.xlim([-6, 6])
    # ax.xlim([0, np.pi])
    # ax.ylim([0, 10])
    for i in range(len(centers)):
        ax.scatter(centers[i, 1], centers[i, 0], marker='x', c='r')
        ax.add_patch(patches.Rectangle((boxes[i, 1] - boxes[i, 3]/2, boxes[i, 0] - boxes[i, 2]/2),     # top left corner coordinates
                        boxes[i, 3],       # width
                        boxes[i, 2],       # height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'))
        # ax.add_patch(patches.Rectangle((boxes[i, 0], boxes[i, 3]), 0.22, 0.1, 
        #         fill=True, clip_on=False, linewidth=2, color='r'))
        # ax.annotate(f'track {t_list[i].id}', (boxes[i, 0], boxes[i, 3]), color='w', ha='left', va='bottom')
        # cx = centers[i, 0]
        # cy = centers[i, 1]
        # ax.text(cx, cy + 0.5, f' Tr: {t_list[i].id}', fontdict={'color': 'red'})
    # ax.axvline(x=-1.70, linewidth=5, color='k')
    # ax.axvline(x=2.30, linewidth=5, color='k')

    if action == 'save':
        plt.savefig(path + 'fig_{}'.format(index), format='png', dpi=250)
        plt.close()
    elif action == 'plot':
        plt.title('Frame {}'.format(index))
        plt.show()
        plt.close()

if __name__ == '__main__':    
    folder = "final"
    rawpath = f'save/jp/proc/denoised/'
    savepath = f'save/jp/{folder}'
    savename = "truth"
    plotpath = "figs"

    # Create the subsequent save folders
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    for i, fname in enumerate(os.listdir(rawpath)):
        frawname = f'ra_{fname.split(".")[0].split("_")[1]}_{fname.split(".")[0].split("_")[3]}'
        print(f'{i+1} / {len(os.listdir(rawpath))} {frawname}')
        # exit()
        rc('text', usetex=True)
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.sans-serif'] = ['Times']
        # the path to the RDA folder to use has to be specified in the arguments
        # args = parse_args()
        # starting index in the loaded data
        start = 10
        # load RDA data, MUST have 4D shape: (N_range_bins, N_angle_bins, N_doppler_bins, N_timesteps)
        rda_data = np.load(f'{rawpath}/{fname}')[..., start:]
        # print(rda_data.shape)
        # path where to save the resulting figures
        # initialize clustering/tracker parameters
        MAX_AGE = 10
        MIN_DET_NUMBER = 15
        MIN_PTS_THR = 30
        MIN_SAMPLES = 40
        EPS = 0.04
        thr = 20
        assoc_score = 'Mahalanobis'                      # either 'IOU' or 'Mahalanobis'
        CLASS_CONF_THR = 0.0
        # init radar parameters
        c0 = 1/np.sqrt(4*np.pi*1e-7*8.85e-12)
        f_start = 76e9
        f_stop = 78e9
        Tramp_up = 180e-6
        Tramp_down = 32e-6
        Tp = 250e-6
        T_int = 66.667e-3
        N = 512
        N_loop = 256
        Tx_power = 100
        kf = 1.1106e13
        BrdFuSca = 4.8828e-5
        fs = 2.8571e6
        fc = (f_start + f_stop)/2
        # compute range angle doppler intervals
        NFFT = 2**10
        nr_chn = 16
        v_range = np.arange(NFFT)/NFFT*fs*c0/(2*kf)
        r_min = 0.5
        r_max = 10
        arg_rmin = np.argmin(np.abs(v_range - r_min))
        arg_rmax = np.argmin(np.abs(v_range - r_max))
        vrange_ext = v_range[arg_rmin:arg_rmax+1]
        NFFT_ant = 64
        vang_deg = np.arcsin(2*np.arange(-NFFT_ant/2, NFFT_ant/2)/NFFT_ant)/np.pi*180
        NFFT_vel = 256
        vfreq_vel = np.arange(-NFFT_vel/2, NFFT_vel/2)/NFFT_vel*(1/Tp)
        v_vel = vfreq_vel*c0/(2*fc)

        delta_r = vrange_ext[1] - vrange_ext[0]
        delta_v = v_vel[1] - v_vel[0]
        delta_a = vang_deg[1] - vang_deg[0]

        action = 'save'

        track_id_list = list(range(1000))   # list with possible track id numbers
        tracking_list = []

        # loop over the time-steps
        for timestep in range(rda_data.shape[-1]):
            print('Timestep {t}'.format(t=timestep+1), end='\r')
            # select RDA map of the current time-step
            data = rda_data[:, :, :, timestep]

            # plt.imshow(data.max(2))
            # plt.show()

            # compute normalized maps for DBSCAN
            norm_ang = (vang_deg - np.min(vang_deg)) / (np.max(vang_deg) - np.min(vang_deg))
            norm_vel = (v_vel - np.min(v_vel)) / (np.max(v_vel) - np.min(v_vel))
            norm_ran = (vrange_ext - np.min(vrange_ext)) / (np.max(vrange_ext) - np.min(vrange_ext))

            rav_pts = np.asarray(np.meshgrid(vrange_ext, vang_deg, v_vel,  indexing='ij'))
            norm_rav_pts = np.asarray(np.meshgrid(norm_ran, norm_ang, norm_vel,  indexing='ij'))

            # select values which are over the threshold
            data = data[arg_rmin:arg_rmax + 1]
            full_indices = (data > thr)
            data[data < thr] = 0

            # plt.imshow(data.max(2))
            # plt.colorbar()
            # plt.show()

            rav_pts = rav_pts[:, full_indices]
            power_values_full = data[full_indices]
            norm_rav_pts = norm_rav_pts[:, full_indices]
            rav_pts_lin = rav_pts.reshape(rav_pts.shape[0], -1)
            # save range and angle for tracking
            ra_totrack = np.copy(rav_pts_lin[:2, :])
            ra_totrack[1] = deg2rad_shift(ra_totrack[1])

            normrav_pts_lin = norm_rav_pts.reshape(norm_rav_pts.shape[0], -1)

            to_cartesian = np.stack([rav_pts[0]*np.cos(rav_pts[1]), rav_pts[0]*np.sin(rav_pts[1])], 0)
            # print(to_cartesian.shape)
            idx = np.logical_not((to_cartesian[0] > -1.70)*(to_cartesian[0] < 2.30))

            ra_totrack = ra_totrack[..., idx]
            rav_pts_lin = rav_pts_lin[..., idx]
            normrav_pts_lin = normrav_pts_lin[..., idx]
            power_values_full = power_values_full[..., idx]

            if rav_pts.shape[1] > MIN_SAMPLES:
                # apply DBSCAN on normalized RDA map
                labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(normrav_pts_lin.T)
                unique, counts = np.unique(labels, return_counts=True)
                if not len(unique):
                    print('DBSCAN found no clusters! Skipping frame.')
                    continue
            else:
                print('No points to cluster! Skipping frame.')
                continue

            # loop over the detected clusters 
            detected_clusters = []   # list containing all the detected clusters
            for cluster_id in unique:
                if cluster_id == -1:  # -1 is the label for noise in DBSCAN, skip it
                    continue
                number = counts[unique == cluster_id]
                if number < MIN_PTS_THR:
                    continue            
                # initialize new cluster object and fill its fields
                new_cluster = Cluster(cluster_id)
                new_cluster.cardinality = number
                new_cluster.elements = ra_totrack[:, labels == cluster_id]   # range and angle
                new_cluster.dopplers = rav_pts_lin[2, labels == cluster_id]

                w = np.squeeze(power_values_full[labels == cluster_id])
                weights = w/np.sum(w)   # normalized powers
                new_cluster.center_polar = np.average(new_cluster.elements, weights=weights, axis=1).reshape(2, 1)
                new_cluster.center_cartesian = np.array([new_cluster.center_polar[0]*np.cos(new_cluster.center_polar[1]), 
                                                        new_cluster.center_polar[0]*np.sin(new_cluster.center_polar[1])], dtype=np.float64).reshape(-1, 1)
                new_cluster.box = get_box(new_cluster)
                detected_clusters.append(new_cluster)

            if not timestep:    # happens only in the first time-step
                for cl in detected_clusters:
                    tracking_list.append(KalmanTracker(id_=track_id_list.pop(0), s0=np.array([cl.center_cartesian[0], 0, cl.center_cartesian[1], 0], 
                                                    dtype=np.float64).reshape(-1,1)))
                    tracking_list[-1].box = cl.box                 
                sel_tracking_list = np.copy(tracking_list)

            elif timestep:    # happens in all other time-steps
                # prepare the data association building the cost matrix
                detected_centers = [x.center_cartesian for x in detected_clusters]
                prev_cartcenters = []
                prev_centers = []
                if len(tracking_list) > 0:
                    for trk in tracking_list:
                        prev_cartcenters.append(trk.xy)
                        prev_centers.append(trk.rtheta)
                cost_matrix = np.zeros((len(detected_centers), len(prev_cartcenters)))
                for i in range(len(detected_centers)):
                    for j in range(len(prev_cartcenters)):
                        # cost is the Mahalanobis distance
                        cost_matrix[i, j] = KalmanTracker.get_mahalanobis_distance(detected_centers[i] - prev_cartcenters[j], tracking_list[j].get_S())  
                cost_matrix = np.asarray(cost_matrix)

                # hungarian algorithm for track association
                matches, undet, unmatch = KalmanTracker.hungarian_assignment(cost_matrix)

                # handle matched tracks
                if len(matches) > 0:
                    for detec_idx, track_idx in matches:
                        # get observation, polar coords center of the detected cluster
                        obs = detected_clusters[detec_idx].center_polar
                        # get tracker object of the detection
                        current_tracker = tracking_list[track_idx]
                        # KF predict-update step
                        current_tracker.predict()
                        current_tracker.update(obs.reshape(2, 1))
                        current_tracker.box = get_box(detected_clusters[detec_idx])
                        current_tracker.hits += 1
                        current_tracker.misses_number = 0 
                        # imaging(current_tracker, detected_clusters[detec_idx], data, labels, full_indices.ravel())
                else:
                    print('No detections-tracks matches found! Skipping frame.')
                    continue
    
                # # deal with unmatched detections
                # if len(unmatch) > 0:
                #     for idx in unmatch:
                #         unmatch_scores = cost_matrix[idx, :]
                #         newc =  detected_clusters[idx].center_cartesian
                #         new_tracker = KalmanTracker(id_=track_id_list.pop(0), s0=np.array([newc[0], 0, newc[1], 0], dtype=np.float64).reshape(-1, 1))
                #         new_tracker.predict()
                #         new_tracker.box = get_box(detected_clusters[idx])
                #         tracking_list.append(new_tracker)

                # deal with undetected tracks
                if len(undet) > 0:
                    for track_idx in undet:
                        old_tracker = tracking_list[track_idx]
                        old_tracker.misses_number += 1
                        # predict only as no obs is detected
                        old_tracker.predict()
                        old_tracker.box = get_box(None, 
                                                  c=old_tracker.xy, 
                                                  h=old_tracker.box[0],
                                                  w=old_tracker.box[0])
                # filter out tracks outside room borders (ghost targets)
                tracking_list = [t for t in tracking_list if (t.xy[0] > -1.70) and (t.xy[0] < 2.30)]    # kill tracks outside the room boundaries
                # select the valid tracks, i.e., the ones with less than the max. misses and enough hits
                sel_tracking_list = [t for t in tracking_list if (t.misses_number <= MAX_AGE) and (t.hits >= MIN_DET_NUMBER)]
                # continue
            
            # print(ra_totrack.shape, data.shape, len(labels))
            # finalname = f'{savepath}/{frawname}_{timestep+1}.npy'
            # rav_pts[1] = deg2rad_shift(rav_pts[1])
            # finalsave = np.insert(rav_pts.T, 3, labels, axis=1)
            # np.save(finalname, finalsave)
            plot(f'{plotpath}/{frawname}_', data, sel_tracking_list, labels, action, timestep)
            # exit()

