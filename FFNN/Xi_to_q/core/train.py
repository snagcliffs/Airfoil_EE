import numpy as np
from tqdm import tqdm
import argparse
import subprocess

import tensorflow as tf
tfk = tf.keras
tfk.backend.set_floatx('float64')

np.random.seed(0)
tf.random.set_seed(0)

from Xi_to_q_generator import Xi_to_q_generator
from Xi_to_q_net import Xi_to_q_net

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main(args):
    """
    
    """

    # Make sure we have folders to save in
    subprocess.run('mkdir '+args.save_path, shell=True)
    subprocess.run('mkdir '+args.save_path+'temp_files', shell=True)

    ##
    ## Set up data generator
    ##
    gen = Xi_to_q_generator(args.data_path, 
                             args.tau, 
                             m_hist = args.m_hist, 
                             stride = args.stride,  
                             batch_size=args.batch_size)
    
    ##
    ## Set up NN parameters
    ##
    data_params = [args.m_hist,32]

    layer_sizes = [args.input_dense_layer_sizes, args.lstm_size, args.output_dense_layer_sizes]
    net_params = [layer_sizes,
                  args.activation]
    
    if args.decay_steps is None: decay_steps = 2*gen.train_batches
    else: decay_steps = decay_steps

    learning_params = [args.l1_reg,
                       args.l2_reg,
                       args.lr,
                       decay_steps,
                       args.decay_rate]
    
    ##
    ## Loop over n_restarts
    ##

    Models = []
    Val_losses = []

    for trial in range(args.n_restarts):

        print("#\n#\n#\n#\n#")
        print("Tau="+str(args.tau)+", trial "+str(trial+1)+" of "+str(args.n_restarts))
        print("#\n#\n#\n#\n#")

        Models.append(Xi_to_q_net(data_params,net_params,learning_params))
        Models[-1].train_model(args.epochs, gen, args.patience, args.save_path+'temp_files/'+args.save_file)
        Val_losses.append(np.min(Models[-1].val_loss))

    best_model = Models[np.argmin(Val_losses)]
    best_model.save_network(args.save_path+args.save_file+'_tau'+str(args.tau))

    # Also save arrays of q_true and q_hat for plotting.  
    q_true = gen.q 
    q_hat = best_model.predict_full_data(gen)
    results = {'true' : q_true, 'NN' : q_hat}
    np.save(args.save_path+'results_tau'+str(args.tau), results)

if __name__ == "__main__":
    """

    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='../../../Re_17500/', type=str, help='Path to datasets')
    parser.add_argument('--save_path', default='../saved_models/', type=str, help='Path to save results')
    parser.add_argument('--save_file', default='Xi_q_LSTM', type=str, help='File name for saved results')

    # Network structure
    parser.add_argument('--input_dense_layer_sizes', type=int, nargs='+', default=[32], help='Pre-LSTM dense layers')
    parser.add_argument('--lstm_size', default=[32,32], type=int, nargs='+', help='Units in LSTM layer')
    parser.add_argument('--output_dense_layer_sizes', type=int, nargs='+', default=[32,16,8,4], help='Post-LSTM dense layers')
    parser.add_argument('--activation', type=str, default='swish', help='Encoder dense layers activation')

    # NN input parameters
    parser.add_argument('--m_hist', default=70, type=int, help='Number of history points to use in training')
    parser.add_argument('--stride', default=3, type=int, help='Spacing of history points')

    # Lead time
    parser.add_argument('--tau', default=5.0, type=float, help='Lead time.')

    # NN training details
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=0.95, type=int, help='Decay rate')
    parser.add_argument('--decay_steps', default=None, help='Decay stesps')
    parser.add_argument('--l1_reg', default=0, type=int, help='L1 penalty on NN weights')
    parser.add_argument('--l2_reg', default=0, type=int, help='L2 penalty on NN weights')
    parser.add_argument('--n_restarts', default=3, type=int, help='Number of restarts.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size')
    parser.add_argument('--patience', default=3, type=int, help='Optimization patience.')

    parser.add_argument('--verbose', type=int, default=0, help='Verbosity of program (not currently used)')
    parser.add_argument('--noise_percent', default=0, type=int, help='Added noise (not currently used)')

    args = parser.parse_args()
    main(args)

