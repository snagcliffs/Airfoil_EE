import numpy as np
from tqdm import tqdm
import argparse
import subprocess

import tensorflow as tf
tfk = tf.keras
tfk.backend.set_floatx('float64')

np.random.seed(0)
tf.random.set_seed(0)

from Psi_generator import Psi_Generator
from Psi_net import Psi_net
from compute_dense_psi import compute_Psi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # comment to use GPU

def main(args):
    """
    
    """

    # Make sure we have folders to save in
    subprocess.run('mkdir '+args.save_path, shell=True)
    subprocess.run('mkdir '+args.save_path+'temp_files', shell=True)

    ##
    ## Set up data generator
    ##
    S = int(100 / args.n_sensors) * np.arange(args.n_sensors)
    
    gen = Psi_Generator(args.data_path, 
                       '../../POD_files/',
                       args.rank,
                       S=S, 
                       m_hist=args.m_hist, 
                       stride=args.stride, 
                       batch_size=args.batch_size)

    ##
    ## Set up NN parameters
    ##
    data_params = [args.n_sensors,
                   args.m_hist,
                   args.rank]

    layer_sizes = [args.input_dense_layer_sizes, args.lstm_size, args.output_dense_layer_sizes]
    net_params = [layer_sizes, 
                  args.activation]

    # If no decay steps is passed in, set to two epochs
    if args.decay_steps is None: decay_steps = 20*gen.train_batches
    else: decay_steps = args.decay_steps

    learning_params = [args.l1_reg,
                       args.l2_reg,
                       args.lr,
                       decay_steps,
                       args.decay_rate]

    ##
    ## Loop over n_restarts and save best model (measured by val set error)
    ##
    Models = []
    Val_losses = []

    for trial in range(args.n_restarts):

        print("#\n#\n#\n#\n#")
        print("Dim="+str(args.rank)+", Trial "+str(trial+1)+" of "+str(args.n_restarts))
        print("#\n#\n#\n#\n#")

        Models.append(Psi_net(data_params,net_params,learning_params))
        Models[-1].train_model(args.epochs, gen, args.patience, args.save_path+'temp_files/'+args.save_file)
        Val_losses.append(np.min(Models[-1].val_loss))

    best_model = Models[np.argmin(Val_losses)]
    best_model.save_network(args.save_path+args.save_file+'_'+str(args.rank))

    # Save Xi(t) sampled on a finer grid than saved velocity files
    compute_Psi(best_model, args.data_path, '../dense_Psi_predictions/', args.stride)

if __name__ == "__main__":
    """

    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='../../../Re_17500/', type=str, help='Path to datasets')
    parser.add_argument('--save_path', default='../saved_models/', type=str, help='Path to save results')
    parser.add_argument('--save_file', default='P_to_Psi', type=str, help='File name for saved results')

    # Network structure
    parser.add_argument('--input_dense_layer_sizes', type=int, nargs='+', default=[64], help='Pre-LSTM dense layers')
    parser.add_argument('--lstm_size', default=[128], type=int, help='Units in LSTM layer')
    parser.add_argument('--output_dense_layer_sizes', type=int, nargs='+', default=[64], help='Post-LSTM dense layers')
    parser.add_argument('--rank', type=int, default=32, help='Latent space rank')
    parser.add_argument('--activation', type=str, default='swish', help='Encoder dense layers activation')

    # NN input parameters
    parser.add_argument('--n_sensors', default=50, type=int, help='Number of sensors')
    parser.add_argument('--m_hist', default=70, type=int, help='Number of history points to use in training')
    parser.add_argument('--stride', default=3, type=int, help='Spacing of history points')

    # NN training details
    parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=1, type=int, help='Decay rate')
    parser.add_argument('--decay_steps', default=None, type=int, help='Decay stesps')
    parser.add_argument('--l1_reg', default=0, type=int, help='L1 penalty on NN weights')
    parser.add_argument('--l2_reg', default=0, type=int, help='L2 penalty on NN weights')
    parser.add_argument('--n_restarts', default=3, type=int, help='Number of restarts.')
    parser.add_argument('--epochs', default=10000, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--patience', default=20, type=int, help='Optimization patience.')

    parser.add_argument('--verbose', type=int, default=0, help='Verbosity of program (not currently used)')
    parser.add_argument('--noise_percent', default=0, type=int, help='Added noise (not currently used)')

    args = parser.parse_args()
    main(args)

