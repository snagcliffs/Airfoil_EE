import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Psi_gen import Psi_gen
from LSTM_ROM import LSTM_ROM

import sys
sys.path.append('../../Psi_to_q/core')
from Psi_to_q_net import Psi_to_q_net

np.random.seed(0)
tf.random.set_seed(0)

def LSTM_q_forecast(H, Q, Psi0, n_steps, Psi_mean, Psi_std):
    
    Psi_est = []
    q_forecast = []
    
    for step in range(n_steps):
        Psi_est.append(tf.expand_dims(H.network(Psi0), axis=1))
        q_forecast.append(Q.network(Psi_est[-1][:,0,:]*Psi_std + Psi_mean).numpy())
        Psi0 = tf.concat([Psi0[:,1:,:], Psi_est[-1]], axis=1)
    
    q_forecast = np.stack(q_forecast,axis=1)
        
    return q_forecast

def main():

    # Load model for Psi to q without forecast
    Q = Psi_to_q_net(restart_file='../../Psi_to_q/saved_models/Psi_q_NN_tau0.0.npy')

    # Load LSTM dynamic model for Psi
    H = LSTM_ROM(restart_file='../saved_models/Psi_ROM_32.npy')

    # Load Psi from file to get normalization used by LSTM model
    Psi = np.load('../../P_to_Psi/dense_Psi_predictions/Psi_32.npy', allow_pickle=True).item()['Psi']
    Psi_mean = np.mean(Psi, axis=0)
    Psi_std = np.std(Psi, axis=0)

    # Set up data generator
    r = 32
    n_steps = 1
    batch_size = 100
    m_hist = 70
    stride = 3
    gen = Psi_gen(r, n_steps, batch_size, m_hist, stride)
    
    ROM_Results = {}
    for j in range(16): ROM_Results[np.round(0.7*j,1)] = {'true' : [], 'NN' : []}
    
    forecast_steps = int(1050 / stride)

    for j in tqdm(range(int(np.ceil(gen.m/batch_size)))):
            
        inds = list(np.arange(batch_size*j, np.min([gen.m,batch_size*(j+1)])))
        Psi0 = gen.get_batch(inds)[0] # Initial condition
        q_f = LSTM_q_forecast(H, Q, Psi0, forecast_steps, Psi_mean, Psi_std)
        
        for i in range(16):
            
            key = np.round(0.7*i,1)
            inds_ = [k for k in inds if k+70*i < gen.m]

            ROM_Results[key]['true'] = ROM_Results[key]['true'] + [gen.q[k+int(70*i)] for k in inds_]
            
            if i > 0:
                ROM_Results[key]['NN'] = ROM_Results[key]['NN'] + \
                                         [q_f[k,int(70*i/stride)-1] for k in range(len(inds_))]
                
            else:
                q0 = Q.network(Psi0[:,-1,:]).numpy()
                ROM_Results[key]['NN'] = ROM_Results[key]['NN'] + list(q0)

    np.save('../saved_models/Psi_ROM_results', ROM_Results)

if __name__ == "__main__":
    """

    """

    main()
