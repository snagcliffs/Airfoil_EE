import numpy as np
from tqdm import tqdm
import argparse
import subprocess

import tensorflow as tf
tfk = tf.keras
tfk.backend.set_floatx('float64')

np.random.seed(0)
tf.random.set_seed(0)

from FFNN_generator import FFNN_Generator
from FFNN_net import pressure_encoder
from compute_dense_xi import compute_Xi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # comment to use GPU

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
    dist_func = lambda d : (1-args.eps)/(1+np.exp(((d-args.l)/args.delta).clip(-np.inf, 35))) + args.eps
    
    gen = FFNN_Generator(args.data_path, 
         dist_func,
         S=S, 
         m_hist=args.m_hist, 
         stride=args.stride, 
         batch_size=args.batch_size, 
         ind_min = args.ind_min)


    ##
    ## Set up NN parameters
    ##
    data_params = [args.n_sensors,
                   gen.n,
                   args.m_hist,
                   args.latent_rank,
                   2]

    encoder_layer_sizes = [args.input_dense_layer_sizes, args.lstm_size, args.output_dense_layer_sizes]
    net_params = [gen.loc_weights, 
                  encoder_layer_sizes, 
                  args.decoder_layer_sizes, 
                  args.activation, 
                  args.residual]

    # If no decay steps is passed in, set to two epochs
    if args.decay_steps is None: decay_steps = 2*gen.train_batches
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
        print("Dim="+str(args.latent_rank)+", Trial "+str(trial+1)+" of "+str(args.n_restarts))
        print("#\n#\n#\n#\n#")

        Models.append(pressure_encoder(data_params,net_params,learning_params))
        Models[-1].train_model(args.epochs, gen, args.patience, args.save_path+'temp_files/'+args.save_file)
        Val_losses.append(np.min(Models[-1].val_loss))

    best_model = Models[np.argmin(Val_losses)]
    best_model.save_network(args.save_path+args.save_file+'_'+str(args.latent_rank))

    # Save Xi(t) sampled on a finer grid than saved velocity files
    compute_Xi(best_model, args.data_path, '../dense_Xi_predictions/', args.stride)

if __name__ == "__main__":
    """

    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='../../../Re_17500/', type=str, help='Path to datasets')
    parser.add_argument('--save_path', default='../saved_models/', type=str, help='Path to save results')
    parser.add_argument('--save_file', default='FFNN', type=str, help='File name for saved results')

    # Network structure
    parser.add_argument('--input_dense_layer_sizes', type=int, nargs='+', default=[64], help='Pre-LSTM dense layers')
    parser.add_argument('--lstm_size', default=[64], type=int, help='Units in LSTM layer')
    parser.add_argument('--output_dense_layer_sizes', type=int, nargs='+', default=[64], help='Post-LSTM dense layers')
    parser.add_argument('--latent_rank', type=int, default=32, help='Latent space rank')
    parser.add_argument('--decoder_layer_sizes', type=int, nargs='+', default=[64,128,256], help='Decoder layer sizes')
    parser.add_argument('--activation', type=str, default='swish', help='Encoder dense layers activation')
    parser.add_argument('--residual', type=bool, default=True, help='Use residual connections in decoder.')

    # NN input parameters
    parser.add_argument('--n_sensors', default=50, type=int, help='Number of sensors')
    parser.add_argument('--m_hist', default=70, type=int, help='Number of history points to use in training')
    parser.add_argument('--stride', default=3, type=int, help='Spacing of history points')
    parser.add_argument('--ind_min', default=80, type=int, help='Minimal timestep used for training')

    # NN loss function
    parser.add_argument('--l', default=1.0, type=float, help='Characteristic width of loss focul region.')
    parser.add_argument('--delta', default=0.1, type=float, help='Characteristic width of sigmoid transition.')
    parser.add_argument('--eps', default=0.1, type=float, help='Weight of error in far field.')

    # NN training details
    parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=0.95, type=int, help='Decay rate')
    parser.add_argument('--decay_steps', default=None, type=int, help='Decay stesps')
    parser.add_argument('--l1_reg', default=1e-3, type=int, help='L1 penalty on NN weights')
    parser.add_argument('--l2_reg', default=1e-3, type=int, help='L2 penalty on NN weights')
    parser.add_argument('--n_restarts', default=1, type=int, help='Number of restarts.')
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--patience', default=5, type=int, help='Optimization patience.')

    parser.add_argument('--verbose', type=int, default=0, help='Verbosity of program (not currently used)')
    parser.add_argument('--noise_percent', default=0, type=int, help='Added noise (not currently used)')

    args = parser.parse_args()
    main(args)

