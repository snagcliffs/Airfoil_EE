import numpy as np
import os
from scipy.signal import gaussian
from scipy.ndimage import filters
from numba import njit
from tqdm import tqdm
"""
To do:
n/a

"""

class P_to_EE_generator():
    """

    """
    
    def __init__(self, 
                 data_path='../Re_17500/', 
                 tau=0, 
                 batch_size=10, 
                 min_time=70, 
                 m_hist=1, 
                 stride=1,
                 train_frac=0.7,
                 val_frac=0.15,
                 contiguous_sets='all'):

        self.data_path = data_path
        self.tau = tau
        self.batch_size = batch_size
        self.min_time = min_time
        self.m_hist = m_hist
        self.stride = stride

        self.train_frac = train_frac
        self.val_frac = val_frac
        assert self.train_frac + self.val_frac < 1
        self.contiguous_sets = contiguous_sets

        # Load grid locations, snapshot times, and distance to wall boundary
        P = np.load(data_path + 'P_reg.npy')
        min_ind = np.min(np.where(P[:,0] > self.min_time))-(self.m_hist-1)*self.stride
        
        if self.tau != 0:
            max_ind = np.min(np.where(P[:,0] > np.max(P[:,0])-self.tau))
        else:
            max_ind = len(P[:,0])

        S = 2*np.arange(50)+1
        self.P = P[min_ind:max_ind,S]
        self.P_time = P[min_ind:max_ind,0]

        # Size of data (subtract (m_hist-1)*stride for NN inputs)
        self.m, self.r = self.P.shape
        self.rnn_input_len = (self.m_hist-1)*self.stride
        self.m = self.m - self.rnn_input_len

        # Load force data
        print('loading force coeffs')
        self.load_force_coeffs()
        print('done')

        # Split dataset into train/val/test
        self.split_dataset()
        self.train_queue = np.random.permutation(self.train_inds)
        self.val_queue = np.random.permutation(self.val_inds)
        self.test_queue = np.random.permutation(self.test_inds)

    def load_force_coeffs(self):

        # Load force coefficients
        forceCoeffs = np.load(self.data_path+'forceCoeffs_reg.npy')
        fc_time = forceCoeffs[100:,0]
        Cd = forceCoeffs[100:,1]
        Cl = forceCoeffs[100:,2]
        dt = fc_time[1]-fc_time[0]
        tau_steps = int(self.tau / dt)

        # Get peak frequency of drag coefficient and set up smoother
        F_Cd = np.fft.fft(Cd - np.mean(Cd))
        freqs = np.fft.fftfreq(len(Cd), d=dt)
        f_peak = freqs[np.argmax(np.abs(F_Cd))]

        width_smoother = int(3/(f_peak*dt))
        scale_smoother = int(0.5/(f_peak*dt))
        smoother_kern = gaussian(width_smoother, scale_smoother)
        smoother_kern = smoother_kern/np.sum(smoother_kern)

        # Quantity of interest as smoothed Cd
        inds = np.argmin(np.abs(self.P_time[0] + self.tau - fc_time)) + np.arange(self.m + (self.m_hist-1)*self.stride)
        self.q = filters.convolve1d(Cd, smoother_kern)
        self.q = (self.q - np.mean(self.q))/np.std(self.q)
        self.q = self.q[inds].reshape(len(inds),1)

    def split_dataset(self):

        self.m_train = int(self.m*self.train_frac)
        self.m_val = int(self.m*self.val_frac)
        self.m_test = self.m - self.m_train - self.m_val

        if self.contiguous_sets == 'all':
            """
            train, val, and test all contiguous
            test will be separated from train by val.
            """
            self.train_inds = np.arange(self.m_train)
            self.val_inds = self.m_train + np.arange(self.m_val)
            self.test_inds = self.m_train + self.m_val + np.arange(self.m_test)

        elif self.contiguous_sets == 'test':
            """
            Train and val mixed up, test is one contiguous set
            """
            self.train_inds = np.random.choice(self.m_train+self.m_val,self.m_train,replace=False)
            self.val_inds = np.array([j for j in np.arange(self.m_train+self.m_val) if j not in self.train_inds])
            self.test_inds = self.m_train + self.m_val + np.arange(self.m_test)

        elif self.contiguous_sets == 'none':
            """
            All datasets randomly mixed
            """
            self.train_inds = np.random.choice(self.m,self.m_train,replace=False)
            self.val_inds = np.random.choice([j for j in np.arange(self.m) if j not in self.train_inds], self.m_val, replace=False)
            self.test_inds = np.array([j for j in range(self.m) if j not in self.train_inds and j not in self.val_inds])

        else:
            raise Exception('contiguous_sets option not recognized')

        # Shift all indexes to account for RNN input
        self.train_inds = self.train_inds + self.rnn_input_len
        self.val_inds = self.val_inds + self.rnn_input_len
        self.test_inds = self.test_inds + self.rnn_input_len

        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))

    def batches_per_epoch(self):

        return self.train_batches, self.val_batches, self.test_batches

    def next_train(self):

        batch_inds = self.train_queue[:self.batch_size]
        self.train_queue = self.train_queue[self.batch_size:]
        if len(self.train_queue) == 0: self.train_queue = np.random.permutation(self.train_inds)

        return self.get_batch(batch_inds)

    def next_val(self):

        batch_inds = self.val_queue[:self.batch_size]
        self.val_queue = self.val_queue[self.batch_size:]
        if len(self.val_queue) == 0: self.val_queue = np.random.permutation(self.val_inds)

        return self.get_batch(batch_inds)

    def next_test(self):

        batch_inds = self.test_queue[:self.batch_size]
        self.test_queue = self.test_queue[self.batch_size:]
        if len(self.test_queue) == 0: self.test_queue = np.random.permutation(self.test_inds)

        return self.get_batch(batch_inds)

    @staticmethod
    @njit
    def get_P_hist(P, batch_inds, m_hist, stride):

        P_hist_batch = np.zeros((len(batch_inds), m_hist, P.shape[1]))

        for i in range(len(batch_inds)):
            for j in range(m_hist):
                for k in range(P.shape[1]):
                    P_hist_batch[i,m_hist-j-1,k] = P[batch_inds[i]-j*stride, k]

        return P_hist_batch

    def get_batch(self, batch_inds):
        """

        """

        if self.m_hist > 1:
            P_batch = self.get_P_hist(self.P, batch_inds, self.m_hist, self.stride)
        else:
            P_batch = self.P[batch_inds,...]

        q_batch = self.q[batch_inds,...]

        return P_batch, q_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))


