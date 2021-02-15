import os
import shutil
from dataclasses import dataclass, field
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass
class Cluster:
    # cluster object, contains detected cluster points and additional values
    label: int 
    cardinality: int = 0
    elements: List = field(default_factory=list)
    dopplers: List = field(default_factory=list)
    center_polar: np.ndarray = np.empty((2, 1))
    center_cartesian: np.ndarray = np.empty((2, 1))
    box: np.ndarray = np.empty((4, 1))

def polar2cartesian(xp):
    # angles in rad
    return np.array([xp[0]*np.cos(xp[1]), xp[0]*np.sin(xp[1])], dtype=np.float64).reshape(-1, 1)

def cartesian2polar(xy):
    # angles in rad
    return np.array([np.sqrt(xy[0]**2 + xy[1]**2), np.arctan2(xy[1], xy[0])]).reshape(-1, 1)

def deg2rad_shift(angles):
    a = np.copy(angles)
    a = np.pi*a/180
    a = -a + np.pi/2
    return a

def shift_rad2deg(angles):
    a = np.copy(angles)
    a = -a + np.pi/2
    a = 180*a/np.pi
    return a

def get_box(cluster, c=None, h=0.5, w=0.3):
    if cluster is not None:
        r_ext = cluster.elements[0].max() - cluster.elements[0].min()
        # print(cluster.elements[1])
        a_ext = cluster.elements[1].max() - cluster.elements[1].min()
        out = np.array([cluster.center_polar[0].squeeze(),
                        cluster.center_polar[1].squeeze(),
                        r_ext,
                        a_ext]).reshape(4, 1)
        return out
    else:
        return np.array([c[0], c[1], h, w]).reshape(4, 1)

def IOU_score(a, b):
	# returns the IOU score of the two input boxes
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    width = x2 - x1
    height = y2 - y1
    if (width < 0) or (height < 0):
        return 0.0
    area_intersection = width*height
    area_a = (a[2] - a[0])*(a[3] - a[1])
    area_b = (b[2] - b[0])*(b[3] - b[1])
    area_union = area_a + area_b - area_intersection
    return area_intersection/area_union

# contains bb calculations and plot function for a single bounding box
class Supporter:
    pathTest = ""
    pathInput = ""
    pathRawImages = ""
    
    def __init__(self):
        self.pathTest = "test"
        self.pathInput = "save/jp/final"
        # pathRawImages = "save/raw"

    def main(self):
        for _, fname in enumerate(os.listdir(self.pathInput)):
            input = np.load(f'{self.pathInput}/{fname}')
            print(input.shape)

            # pathCheck(pathTest)
            inputbb = self.label2bb(input)
            fig = self.plotRaw(input, bb=inputbb, save=False, showLabels=True)
            print(type(fig))
                
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            print(data)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            print(data.shape)

            # np.savetxt("test.csv", input, delimiter=",")
            exit()

    #bb: bounding box. show: display figure. showLabels: colorcode the cluster/labels.
    def plotRaw(self, data, bb=None, save=False, show=False, showLabels=False, path=pathTest):
        fig = plt.figure()
        ax = plt.axes()
        plt.axis("off")

        if not showLabels:
            plt.scatter(data[:,0], data[:,1])
        else:
            plt.scatter(data[:,0], data[:,1], c=(data[:,3] == 1))

        if bb != None:
            ax.add_patch(patches.Rectangle((bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], 
                fill=False, clip_on=False, linewidth=2, color='k'))
            ax.add_patch(patches.Rectangle((bb[0], bb[3]), 0.22, 0.1, 
                fill=True, clip_on=False, linewidth=2, color='k'))
            ax.annotate(' Truth', (bb[0], bb[3]), color='w', ha='left', va='bottom')
        if show:
            plt.show()
        if save:
            plt.savefig(f'{path}/_test.png', dpi=100)

        fig.canvas.draw()
        return fig

    #bouding box computation
    def label2bb(self, data):
        xmin = np.min(data[(data[:,3] == 1), 0])
        xmax = np.max(data[(data[:,3] == 1), 0])
        ymin = np.min(data[(data[:,3] == 1), 1])
        ymax = np.max(data[(data[:,3] == 1), 1])
        # print(xmin, xmax, ymin, ymax)
        return [(xmax-xmin)/2, (ymax-ymin)/2, xmax-xmin, ymax-ymin]

    def pathCheck(self, path):
        # Create the subsequent save folders
        if os.path.isdir(path):
            shutil.rmtree(path)
        if not os.path.isdir(path):
            os.makedirs(path)

if __name__ == '__main__':   
    Supporter().main()