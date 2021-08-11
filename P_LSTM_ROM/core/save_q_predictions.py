import numpy as np
import tensorflow as tf
import subprocess
from tqdm import tqdm

from P_gen import P_gen
from LSTM_ROM import LSTM_ROM

import sys
sys.path.append('../../P_to_q')
from P_to_EE_net import P_to_EE_net

np.random.seed(0)
tf.random.set_seed(0)

def LSTM_q_forecast(H, Q, P0, n_steps):
    
    P_est = []
    q_forecast = []
    
    for step in range(n_steps):
        P_est.append(tf.expand_dims(H.network(P0), axis=1))
        q_forecast.append(Q.network(P_est[-1][:,0,:]).numpy())
        P0 = tf.concat([P0[:,1:,:], P_est[-1]], axis=1)
    
    q_forecast = np.stack(q_forecast,axis=1)
        
    return q_forecast

def main():

    # Load model for P to q without forecast
    Q = P_to_EE_net(restart_file='../../P_to_q/saved_results/P_q_NN_tau0.0.npy')

    # Load LSTM dynamic model for Psi
    H = LSTM_ROM(restart_file='../saved_models/P_ROM_50.npy')

    # Set up data generator
    data_path = '../../Re_17500/'
    r = 50
    n_steps = 1
    batch_size = 100
    m_hist = 70
    stride = 3
    gen = P_gen(data_path, r, n_steps, batch_size, m_hist, stride)
    
    ROM_Results = {}
    for j in range(16): ROM_Results[np.round(0.7*j,1)] = {'true' : [], 'NN' : []}
    
    forecast_steps = int(1050 / stride)

    for j in tqdm(range(int(np.ceil(gen.m/batch_size)))):
            
        inds = list(np.arange(batch_size*j, np.min([gen.m,batch_size*(j+1)])))
        P0 = gen.get_batch(inds)[0] # Initial condition
        q_f = LSTM_q_forecast(H, Q, P0, forecast_steps)
        
        for i in range(16):
            
            key = np.round(0.7*i,1)
            inds_ = [k for k in inds if k+70*i < gen.m]

            ROM_Results[key]['true'] = ROM_Results[key]['true'] + [gen.q[k+int(70*i)] for k in inds_]
            
            if i > 0:
                ROM_Results[key]['NN'] = ROM_Results[key]['NN'] + \
                                         [q_f[k,int(70*i/stride)-1] for k in range(len(inds_))]
                
            else:
                q0 = Q.network(P0[:,-1,:]).numpy()
                ROM_Results[key]['NN'] = ROM_Results[key]['NN'] + list(q0)

    np.save('../saved_models/P_ROM_results', ROM_Results)

if __name__ == "__main__":
    """

    """

    main()
