import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import tensorflow as tf
from typing import List
import numpy as np

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

def cartesian2polar(xc):
    # angles in rad
    return np.array([np.sqrt(self.xy[0]**2 + self.xy[1]**2), np.arctan2(self.xy[1], self.xy[0])]).reshape(-1, 1)

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

def get_box(c, h=4, w=8):
    # pass a center to this function to get a box centered there
    return np.concatenate([c + np.array([-h/2, -w/2]).reshape(2, 1),
                           c + np.array([h/2, w/2]).reshape(2, 1)], axis=0)

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



    


