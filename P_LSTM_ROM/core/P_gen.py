import numpy as np
from numba import njit
"""
To do:
n/a

"""

np.random.seed(0)

class P_gen():
    
    def __init__(self, 
                 data_path,
                 r=50,
                 n_steps=10, 
                 batch_size=100, 
                 m_hist=1, 
                 stride=1,
                 train_frac=0.7,
                 val_frac=0.15,
                 contiguous_sets='test',
                 return_q = False):

        self.data_path = data_path
        self.r = r
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.m_hist = m_hist
        self.stride = stride

        self.train_frac = train_frac
        self.val_frac = val_frac
        assert self.train_frac + self.val_frac < 1
        self.contiguous_sets = contiguous_sets

        # Load data
        self.load_P()

        # Size of data (subtract (m_hist-1)*stride for NN inputs)
        self.m, self.r = self.P.shape
        self.rnn_input_len = (self.m_hist-1)*self.stride
        self.m = self.m - self.rnn_input_len - self.n_steps*self.stride

        # Split dataset into train/val/test
        self.split_dataset()
        self.return_q = return_q

    def load_P(self):

        self.S = int(100/self.r)*np.arange(self.r)
        P_T = np.load(self.data_path + 'P.npy')[100:,:]
        self.P = P_T[:,1:][:,self.S]
        self.t = P_T[:,0]

        qt = np.load(self.data_path + 'q.npy')
        q_inds = 10*np.arange(self.P.shape[0])+np.argmin(np.abs(self.t[0] - qt[:,0]))
        self.q = qt[q_inds,1]

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

        # Batches per epoch in each dataset
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))

        # Queues
        self.train_queue = np.random.permutation(self.train_inds)
        self.val_queue = np.random.permutation(self.val_inds)
        self.test_queue = np.random.permutation(self.test_inds)

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

    def next_test(self, return_inds=False):

        batch_inds = self.test_queue[:self.batch_size]
        self.test_queue = self.test_queue[self.batch_size:]
        if len(self.test_queue) == 0: self.test_queue = np.random.permutation(self.test_inds)

        if return_inds: 
            if self.return_q:
                P_hist_batch, P_forecast_batch, q_batch = self.get_batch(batch_inds)
                return P_hist_batch, P_forecast_batch, q_batch, batch_inds

            else:
                P_hist_batch, P_forecast_batch = self.get_batch(batch_inds)
                return P_hist_batch, P_forecast_batch, batch_inds
        else: 
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

    @staticmethod
    @njit
    def get_P_forecast(P, batch_inds, n_steps, stride):

        P_forecast_batch = np.zeros((len(batch_inds), n_steps, P.shape[1]))

        for i in range(len(batch_inds)):
            for j in range(n_steps):
                for k in range(P.shape[1]):
                    P_forecast_batch[i,j,k] = P[batch_inds[i]+(j+1)*stride, k]

        return P_forecast_batch

    @staticmethod
    @njit
    def get_q_forecast(q, batch_inds, n_steps, stride):

        q_forecast_batch = np.zeros((len(batch_inds), n_steps))

        for i in range(len(batch_inds)):
            for j in range(n_steps):
                q_forecast_batch[i,j] = q[batch_inds[i]+(j+1)*stride]

        return q_forecast_batch

    def get_batch(self, batch_inds):
        """
        """
        
        if self.m_hist > 1:
            P_hist_batch = self.get_P_hist(self.P, batch_inds, self.m_hist, self.stride)
        else:
            P_hist_batch = self.P[batch_inds,...]

        P_forecast_batch = self.get_P_forecast(self.P, batch_inds, self.n_steps, self.stride)

        if self.return_q:
            q_batch = self.get_q_forecast(self.q, batch_inds, self.n_steps, self.stride)
            return P_hist_batch, P_forecast_batch, q_batch

        else:
            return P_hist_batch, P_forecast_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))


