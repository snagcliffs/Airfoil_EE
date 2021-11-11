import numpy as np
import tensorflow as tf
import argparse
import subprocess

from Psi_gen import Psi_gen
from LSTM_ROM import LSTM_ROM

np.random.seed(0)
tf.random.set_seed(0)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # comment to use GPU

def main(args):
    """
    
    """

    # Make sure we have folders to save in
    subprocess.run('mkdir '+args.save_path, shell=True)
    subprocess.run('mkdir '+args.save_path+'temp_files', shell=True)

    # Data params
    r = args.r
    n_steps = args.n_steps
    m_hist = args.m_hist
    stride = args.stride
    batch_size = args.batch_size

    gen = Psi_gen(r, n_steps, batch_size, m_hist, stride)

    # Net params
    layer_sizes = [args.input_dense_layer_sizes, 
                   args.lstm_size, 
                   args.output_dense_layer_sizes]
    activation = args.activation

    # Learning params
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    lr = args.lr
    decay_rate = args.decay_rate

    # If no decay steps is passed in, set to two epochs
    if args.decay_steps is None: decay_steps = 2*gen.train_batches
    else: decay_steps = args.decay_steps

    # Training params
    epochs = args.epochs
    patience = args.patience

    data_params = [r, m_hist, stride, n_steps]
    net_params = [layer_sizes, activation]
    learning_params = [l1_reg, l2_reg, lr, decay_steps, decay_rate]
    training_params = [epochs, patience]

    ##
    ## Loop over n_restarts and save best model (measured by val set error)
    ##
    Models = []
    Val_losses = []

    for trial in range(args.n_restarts):

        print("#\n#\n#\n#\n#")
        print("Trial "+str(trial+1)+" of "+str(args.n_restarts))
        print("#\n#\n#\n#\n#")

        Models.append(LSTM_ROM(data_params, net_params, learning_params))
        Models[-1].train_model([args.epochs,args.patience], gen, args.save_path+'temp_files/'+args.save_file)
        Val_losses.append(np.min(Models[-1].val_loss))

    best_model = Models[np.argmin(Val_losses)]
    best_model.save_network(args.save_path+args.save_file+str(r))

if __name__ == "__main__":
    """

    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='../../../Re_17500/', type=str, help='Path to datasets')
    parser.add_argument('--save_path', default='../saved_models/', type=str, help='Path to save results')
    parser.add_argument('--save_file', default='Psi_ROM_', type=str, help='File name for saved results')

    # Network structure
    parser.add_argument('--r', type=int, default=32, help='Dimension of space to learn ROM for [8,16,32,64]')
    parser.add_argument('--input_dense_layer_sizes', type=int, nargs='+', default=[64], help='Pre-LSTM dense layers')
    parser.add_argument('--lstm_size', default=[64,64], type=int, help='Units in LSTM layer')
    parser.add_argument('--output_dense_layer_sizes', type=int, nargs='+', default=[64,32], help='Post-LSTM dense layers')
    parser.add_argument('--activation', type=str, default='swish', help='Encoder dense layers activation')

    # NN input parameters
    parser.add_argument('--m_hist', default=70, type=int, help='Number of history points to use in training')
    parser.add_argument('--stride', default=3, type=int, help='Spacing of history points')

    # NN training details
    parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=0.98, type=int, help='Decay rate')
    parser.add_argument('--decay_steps', default=None, type=int, help='Decay stesps')
    parser.add_argument('--l1_reg', default=0, type=int, help='L1 penalty on NN weights')
    parser.add_argument('--l2_reg', default=0, type=int, help='L2 penalty on NN weights')
    parser.add_argument('--n_restarts', default=3, type=int, help='Number of restarts.')
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=250, type=int, help='Batch size')
    parser.add_argument('--n_steps', default=20, type=int, help='Batch size')
    parser.add_argument('--patience', default=5, type=int, help='Optimization patience.')

    parser.add_argument('--verbose', type=int, default=0, help='Verbosity of program (not currently used)')
    parser.add_argument('--noise_percent', default=0, type=int, help='Added noise (not currently used)')

    args = parser.parse_args()
    main(args)
