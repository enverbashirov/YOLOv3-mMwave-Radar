import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class KalmanTracker:

    def __init__(self, id_, s0=None, disable_rejection_check=False):
        # Filter-related parameters
        self.dt = 66.667e-3    # T_int of the radar TX
        # state transition matrix
        self.F = np.kron(np.eye(2), np.array([[1, self.dt], [0, 1]]))
        # # state-acceleration matrix
        self.G = np.array([0.5*(self.dt**2), self.dt]).reshape(2, 1)
        # # observation matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        # measurement covariance matrix
        self.R = np.array([[0.5, 0], [0, 0.5]])    # [wagner2017radar]
        # initial state covariance
        self.P = 0.2*np.eye(4)
        # state noise variance
        self.sigma_a = 8                           # [wagner2017radar]
        # state noise covariance
        self.Q = np.kron(np.eye(2), np.matmul(self.G, self.G.T)*self.sigma_a**2)
        self.n = self.F.shape[1]
        self.m = self.H.shape[1]
        # initial state
        self.s = np.zeros((self.n, 1)) if s0 is None else s0
        self.xy = np.array([self.s[0], self.s[2]]).reshape(-1, 1)
        self.rtheta = np.array([np.sqrt(self.xy[0]**2 + self.xy[1]**2), np.arctan2(self.xy[1], self.xy[0])]).reshape(-1, 1)
        self.REJECT_THR = 4.605
        self.disable_rejection_check = disable_rejection_check
        ######################################################### 
        # Tracker-related parameters
        self.misses_number = 0
        self.hits = 0
        self.id = id_
        self.box = np.array([])
        self.state_memory = []
        self.identity_label = 'UNK'            # initialize as unknown cluster
        self.id_dict = {-1: 'UNK', 0: 'S1', 1: 'S2', 2:'S3', 3:'S4'}
        # self.id_dict = {-1: 'UNK', 0: 'JP', 1: 'FM', 2:'GP', 3:'RF'}

    def transform_obs(self, z):
        z_prime = np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1])]).reshape(-1, 1)
        return z_prime

    def reject_obs(self, i, S):
        chi_squared = np.matmul(np.matmul(i.T, np.linalg.inv(S)), i)[0, 0]
        return chi_squared >= self.REJECT_THR

    def predict(self):
        a_x = np.random.normal(0, self.sigma_a)
        a_y = np.random.normal(0, self.sigma_a)
        self.s = np.matmul(self.F, self.s) 
        # check that x has the correct shape
        assert self.s.shape == (self.n, 1)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        self.xy = np.array([self.s[0], self.s[2]]).reshape(-1, 1)
        self.rtheta = np.array([np.sqrt(self.xy[0]**2 + self.xy[1]**2), np.arctan2(self.xy[1], self.xy[0])]).reshape(-1, 1)
        return self.s, self.xy

    def update(self, z):
        z = self.transform_obs(z)
        # innovation
        y = z - np.matmul(self.H, self.s)
        S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
        if (not self.reject_obs(y, S)) or self.disable_rejection_check:
            K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(S))
            self.s = self.s + np.matmul(K, y)
            assert self.s.shape == (self.n, 1)
            self.P = np.matmul(np.eye(self.n) - np.matmul(K, self.H), self.P)
            self.xy = np.array([self.s[0], self.s[2]]).reshape(-1, 1)
            self.rtheta = np.array([np.sqrt(self.xy[0]**2 + self.xy[1]**2), np.arctan2(self.xy[1], self.xy[0])]).reshape(-1, 1)
            self.state_memory.append(self.xy)
            return self.s, self.xy
        else:
            self.state_memory.append(self.xy)
            return self.s, self.xy

    def get_S(self):
        return np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
            
    @staticmethod
    def get_mahalanobis_distance(x, C):
        # returns Mahalanobis distance given the differece vector x and covariance C
        return np.matmul(np.matmul(x.T, np.linalg.inv(C)), x)[0, 0]
        
    @staticmethod
    def hungarian_assignment(score_matrix):
        # call the scipy implementation of Hungarian alg.
        det_idx, tr_idx = sp.optimize.linear_sum_assignment(score_matrix)
        unmatched, undetected = [], []
        for t in range(score_matrix.shape[1]):
            if t not in tr_idx:
                undetected.append(t)
        for d in range(score_matrix.shape[0]):
            if d not in det_idx:
                unmatched.append(d)
        matches = []
        for d, t in zip(det_idx, tr_idx):
            matches.append(np.array([d, t]).reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(undetected), np.array(unmatched)

