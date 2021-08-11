import numpy as np
import os

"""
To do:
n/a

"""

class FFNN_Generator():
    """
    Generator object for nonlinear flow reconstruction from pressure sensors.
    May also return single snapshots for visualization.

    Inputs:

        data_path   :  path to saved dataset. We assume there esist files
                       data_path/Cx.npy 
                       data_path/Cy.npy
                       data_path/dist.npy
                       data_path/mass.npy
                       data_path/P.npy
                       data_path/forceCoeffs.npy
                       data_path/numpy_files/vel(index).npy
                       data_path/numpy_files/snapshot_times.npy

        batch_size  :  number of snapshots in each batch
        n_g         :  number of
        S           :  list of indices of pres_hist to use as input
        ind_min     :  number of transient files to skip
        m_hist      :  
        stride      :  
        dist_func   :  
        time_func   :  
        tau         :  lead time used to compute loss weights as a function of time
        train_frac  :  
        val_frac    :  

    Returns:

    """
    
    def __init__(self, 
                 data_path, 
                 dist_func, 
                 time_func = None,
                 tau=0, 
                 batch_size=10, 
                 S=None, 
                 ind_min=200, 
                 m_hist=100, 
                 stride=10,
                 train_frac=0.7,
                 val_frac=0.15,
                 contiguous_sets='test'):

        self.data_path = data_path
        self.dist_func = dist_func
        self.time_func = time_func
        self.tau = tau
        self.batch_size = batch_size
        self.S = S
        self.ind_min = ind_min
        self.m_hist = m_hist
        self.stride = stride

        self.train_frac = train_frac
        self.val_frac = val_frac
        assert self.train_frac + self.val_frac < 1
            
        self.contiguous_sets = contiguous_sets

        # Load grid locations, snapshot times, and distance to wall boundary
        self.X = np.vstack([np.load(data_path + 'Cx.npy'),np.load(data_path + 'Cy.npy')]).T
        self.snapshot_times = np.load(data_path + 'numpy_files/snapshot_times.npy')[self.ind_min:]
        self.dist = np.load(data_path + 'dist.npy')
        self.mass = np.load(data_path + 'mass.npy')

        # Size of data
        self.m = len(self.snapshot_times)
        m2 = len(os.listdir(data_path+'numpy_files'))-1-self.ind_min
        try:
            assert self.m == m2
        except:
            print(self.m)
            print(m2)
            raise Exception('number of velocity files does not match length of time series')
        self.n = self.X.shape[0]

        # Get truncated time series inputs to RNN
        self.compute_P_hist()

        # Get weights for loss function
        self.loc_weights = self.mass*dist_func(self.dist)
        self.loc_weights = self.loc_weights/np.mean(self.loc_weights)
        self.compute_time_weights()

        # Split dataset into train/val/test
        self.split_dataset()
        self.train_queue = list(np.random.permutation(self.train_inds))
        self.val_queue = list(np.random.permutation(self.val_inds))
        self.test_queue = list(np.random.permutation(self.test_inds))

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
            self.val_inds = m_train + np.arange(self.m_val)
            self.test_inds = m_train + m_val + np.arange(self.m_test)

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

        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))

    def batches_per_epoch(self):

        return self.train_batches, self.val_batches, self.test_batches

    def next_train(self):

        batch_inds = self.train_queue[:self.batch_size]
        self.train_queue = self.train_queue[self.batch_size:]
        if len(self.train_queue) == 0: self.train_queue = list(np.random.permutation(self.train_inds))

        return self.get_batch(batch_inds)

    def next_val(self):

        batch_inds = self.val_queue[:self.batch_size]
        self.val_queue = self.val_queue[self.batch_size:]
        if len(self.val_queue) == 0: self.val_queue = list(np.random.permutation(self.val_inds))

        return self.get_batch(batch_inds)

    def next_test(self):

        batch_inds = self.test_queue[:self.batch_size]
        self.test_queue = self.test_queue[self.batch_size:]
        if len(self.test_queue) == 0: self.test_queue = list(np.random.permutation(self.test_inds))

        return self.get_batch(batch_inds)

    def get_batch(self, batch_inds, n_g=None):

        P_batch = self.P_hist[batch_inds,...]
        Vel_batch = np.stack([np.load(self.data_path + 'numpy_files/vel{0:05d}.npy'.format(j+self.ind_min+1)) 
                              for j in batch_inds])

        return P_batch, Vel_batch, self.time_weights[batch_inds]

    def sample(self, m_samp, n_samp):
        """
        Random sample from full dataset
        """
        
        if m_samp > self.m: m_samp = self.m
        if n_samp > self.n: n_samp = self.n

        batch_inds = np.random.choice(self.m, m_samp, replace=False)
        return self.get_batch(batch_inds, n_samp)

    def get_snapshot(self, ind):
        """
        Returns full dataset for index ind.
        Suitable for visualization of comparison between NN reconstruction and truth
        """

        assert ind >=0 and ind < self.m
        vel = np.load(self.data_path + 'numpy_files/vel{0:05d}.npy'.format(ind+self.ind_min+1))
        p_hist = np.expand_dims(self.P_hist[ind,...], axis=0)

        return p_hist, self.X, vel

    def compute_P_hist(self):

        PT = np.load(self.data_path + 'P.npy')
        P_time = PT[:,0]
        P = PT[:,1:]

        if self.S is None: 
            self.S = np.arange(P.shape[1])
        else:
            P = P[:,self.S]

        # Determine indices corresponding to snapshot times
        P_snapshot_inds = [np.argmin(np.abs(P_time - t)) for t in self.snapshot_times]
        P_hist = []

        for j in range(self.m):
            inds = [P_snapshot_inds[j]-i*self.stride for i in range(self.m_hist)]
            P_hist.append(P[inds[::-1],:])

        self.P_hist = np.array(P_hist)
        self.P_time = P_time[P_snapshot_inds]
        if self.m_hist == 1: self.P_hist = self.P_hist.reshape(self.m, self.P_hist.shape[2])

    def compute_time_weights(self):
        """
        Computes weighted loss funciton: r(t) = time_func(q(t+tau))
        """

        if self.time_func is None: 
            self.time_weights = np.ones_like(self.snapshot_times)

        else:   
            tq = np.load(self.data_path + 'q.npy')
            fc_time = tq[10:,0]
            q = tq[10:,1]
            dt = fc_time[1] - fc_time[0]

            # Determine indices corresponding to snapshot times shifted by tau
            # Note that snapshots at time t > T-tau will be weighted by q(T), not q(t+tau)
            fc_snapshot_inds = [np.argmin(np.abs(fc_time - t - self.tau)) for t in self.snapshot_times]
            q_course = q[fc_snapshot_inds]

            self.time_weights = self.time_func(q_course)

    def set_S(self,S):
        self.S = S

    def get_S(self):
        return self.S

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))

    def get_P_hist(self):
        return self.P_hist

    def get_X(self):
        return self.X
