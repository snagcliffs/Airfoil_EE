import numpy as np
from numba import njit
from tqdm import tqdm
"""
To do:
n/a

"""

class Psi_to_q_generator():
    """

    """
    
    def __init__(self, 
                 data_path='../../../Re_17500/', 
                 rank=32,
                 tau=0, 
                 batch_size=10, 
                 min_time=20, 
                 m_hist=1, 
                 stride=1,
                 train_frac=0.7,
                 val_frac=0.15,
                 POD_prefix='midEps', 
                 contiguous_sets='test'):

        self.data_path = data_path
        self.tau = tau
        self.r = rank
        self.batch_size = batch_size
        self.min_time = min_time
        self.m_hist = m_hist
        self.stride = stride

        self.train_frac = train_frac
        self.val_frac = val_frac
        assert self.train_frac + self.val_frac < 1
        self.contiguous_sets = contiguous_sets

        # Load saved Psi
        Psi_dict = np.load('../../P_to_Psi/dense_Psi_predictions/Psi_32.npy', allow_pickle=True).item()
        min_ind = np.min(np.where(Psi_dict['time'] > self.min_time))-(self.m_hist-1)*self.stride
        
        if self.tau != 0:
            max_ind = np.min(np.where(Psi_dict['time'] > np.max(Psi_dict['time'])-self.tau))
        else:
            max_ind = len(Psi_dict['time'])

        self.Psi = Psi_dict['Psi'][min_ind:max_ind,:]
        self.Psi_time = Psi_dict['time'][min_ind:max_ind]

        # Size of data (subtract (m_hist-1)*stride for NN inputs)
        self.rnn_input_len = (self.m_hist-1)*self.stride
        self.m = self.Psi.shape[0] - self.rnn_input_len

        # Load force data
        print('Loading force data')
        self.load_q()

        # Split dataset into train/val/test
        self.split_dataset()
        self.train_queue = np.random.permutation(self.train_inds)
        self.val_queue = np.random.permutation(self.val_inds)
        self.test_queue = np.random.permutation(self.test_inds)

    def load_q(self):

        tq = np.load(self.data_path + 'q.npy')
        fc_time = tq[:,0]
        self.q = tq[:,1]
        dt = fc_time[1]-fc_time[0]
        # inds = list(self.get_inds(fc_time, self.Psi_time, self.tau).astype(int))
        inds = 10*np.arange(len(self.Psi_time)) + np.argmin(np.abs(fc_time - self.Psi_time[0] - self.tau)).astype(int)
        self.q = self.q[inds].reshape(len(inds),1)

    @staticmethod
    @njit
    def get_inds(fc_time, Psi_time, tau):

        inds = np.zeros_like(Psi_time)

        for i in range(len(Psi_time)):
            inds[i] = np.argmin(np.abs(fc_time - Psi_time[i] - tau))

        return inds

    def split_dataset(self):

        self.m_train = int(self.m*self.train_frac)
        self.m_val = int(self.m*self.val_frac)
        self.m_test = self.m - self.m_train - self.m_val

        print('Length of full dataset:', self.m)
        print('Length of training dataset:', self.m_train)
        print('Length of validation dataset:', self.m_val)
        print('Length of testing dataset:', self.m_test)

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
    def get_Psi_hist(Psi, batch_inds, m_hist, stride):

        Psi_hist_batch = np.zeros((len(batch_inds), m_hist, Psi.shape[1]))

        for i in range(len(batch_inds)):
            for j in range(m_hist):
                for k in range(Psi.shape[1]):
                    Psi_hist_batch[i,m_hist-j-1,k] = Psi[batch_inds[i]-j*stride, k]

        return Psi_hist_batch

    def get_batch(self, batch_inds):
        """
        What is different if I want to include history terms here?
        """

        if self.m_hist > 1:
            Psi_batch = self.get_Psi_hist(self.Psi, batch_inds, self.m_hist, self.stride)
        else:
            Psi_batch = self.Psi[batch_inds,...]

        q_batch = self.q[batch_inds,...]

        return Psi_batch, q_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))


