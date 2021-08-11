import numpy as np
from numba import njit
"""
To do:
n/a

"""

np.random.seed(0)

class Psi_gen():
    """

    """
    
    def __init__(self, 
                 r=32,
                 n_steps=10, 
                 batch_size=100, 
                 m_hist=1, 
                 stride=1,
                 train_frac=0.7,
                 val_frac=0.15,
                 contiguous_sets='test',
                 return_q = False):

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
        self.load_Psi()

        # Size of data (subtract (m_hist-1)*stride for NN inputs)
        self.m, self.r = self.Psi.shape
        self.rnn_input_len = (self.m_hist-1)*self.stride
        self.m = self.m - self.rnn_input_len - self.n_steps*self.stride

        # Split dataset into train/val/test
        self.split_dataset()
        self.return_q = return_q

    def load_Psi(self):

        Psi_dict = np.load('../../P_to_Psi/dense_Psi_predictions/Psi_'+str(self.r)+'.npy', allow_pickle=True).item()
        self.Psi = Psi_dict['Psi']
        self.Psi = (self.Psi - np.mean(self.Psi, axis=0))/np.std(self.Psi, axis=0)
        self.q = Psi_dict['q']
        self.t = Psi_dict['time']

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
                Psi_hist_batch, Psi_forecast_batch, q_batch = self.get_batch(batch_inds)
                return Psi_hist_batch, Psi_forecast_batch, q_batch, batch_inds

            else:
                psi_hist_batch, Psi_forecast_batch = self.get_batch(batch_inds)
                return Psi_hist_batch, Psi_forecast_batch, batch_inds
        else: 
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

    @staticmethod
    @njit
    def get_Psi_forecast(Psi, batch_inds, n_steps, stride):

        Psi_forecast_batch = np.zeros((len(batch_inds), n_steps, Psi.shape[1]))

        for i in range(len(batch_inds)):
            for j in range(n_steps):
                for k in range(Psi.shape[1]):
                    Psi_forecast_batch[i,j,k] = Psi[batch_inds[i]+(j+1)*stride, k]

        return Psi_forecast_batch

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
            Psi_hist_batch = self.get_Psi_hist(self.Psi, batch_inds, self.m_hist, self.stride)
        else:
            Psi_hist_batch = self.Psi[batch_inds,...]

        Psi_forecast_batch = self.get_Psi_forecast(self.Psi, batch_inds, self.n_steps, self.stride)

        if self.return_q:
            q_batch = self.get_q_forecast(self.q, batch_inds, self.n_steps, self.stride)
            return Psi_hist_batch, Psi_forecast_batch, q_batch

        else:
            return Psi_hist_batch, Psi_forecast_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))


