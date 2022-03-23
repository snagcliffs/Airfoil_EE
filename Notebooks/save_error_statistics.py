import numpy as np
from guth import *
from tqdm import tqdm
from scipy.signal import gaussian
from scipy.ndimage import filters

def conv_smoother(x, width_smoother=250, scale_smoother=10):
    
    # Gaussian smoother
    smoother_kern = gaussian(width_smoother, scale_smoother)
        
    # Exp decay smoother (only uses history)
    smoother_kern = np.zeros(width_smoother)
    for i in range(int(width_smoother/2)):
        smoother_kern[int(width_smoother/2) + i] = np.exp(-i/scale_smoother)    
            
    smoother_kern = smoother_kern/np.sum(smoother_kern)
    xs = filters.convolve1d(x.flatten(), smoother_kern)
        
    return xs

def compute_Qi_KL(y, z, Tp=1, Tm=1, alpha=0):
    """
    This is similar Eq. 8 from Qi and Majda, PNAS 2020.
    In the paper they use alpha=1, but since we are looking at y,z>0, we use alpha=0
    We are also taking the KL divergence between time series, rather than across dimensions at a single timestep.
    y is prediction
    z is target
    """

    m = y.size

    yp = np.exp(y/Tp) / np.mean(np.exp(y/Tp))
    zp = np.exp(z/Tp) / np.mean(np.exp(z/Tp))
    KLp = np.sum(zp*np.log(zp/yp))/m

    if alpha != 0:
        ym = np.exp(-y/Tm) / np.mean(np.exp(-y/Tm))
        zm = np.exp(-z/Tm) / np.mean(np.exp(-z/Tm))
        KLm = np.sum(zm*np.log(zm/ym))/m
    else: 
        KLm = 0

    return KLp + alpha*KLm

def compute_and_save_errors(Results, save_file, EE_rates=None, smooth=True, m_test=15000):
    """
    The value of m_test=15000 specifies the test set to be [870.01,1020] for all methods.
    """

    if EE_rates is None: EE_rates = np.linspace(0,0.25,26)[1:]
    n_EE = len(EE_rates)

    Taus = list(Results.keys())
    n_tau = len(Taus)

    Q_pred = [Results[tau]['NN'].flatten() for tau in Taus]
    Q_true = [Results[tau]['true'][len(Results[tau]['true'])-len(Results[tau]['NN']):].flatten() for tau in Taus]
    
    if smooth:
        print(Q_pred[0].shape)
        Q_pred = [conv_smoother(q_pred) for q_pred in Q_pred]

    Alpha_star = []
    Q_opt = []
    Qi_KL = []
    F1_opt = []
    MSE = []
    MAE = []

    F1 = np.zeros((n_EE,n_tau))
    AUC = np.zeros((n_EE,n_tau))

    # EE rate independent errors
    for j in tqdm(range(len(Taus))):

        m = len(Q_true[j])
        m_train_val = int(0.85*m)

        # Criterion from Guth 
        alpha_star, q_opt, a_opt, b_opt = guth_criterion(Q_true[j][-m_test:], 
                                                        Q_pred[j][-m_test:],
                                                        return_thresholds=True,
                                                        nq=51,
                                                        q_min=0.0,
                                                        q_max=0.5,
                                                        nb=501)

        Alpha_star.append(alpha_star)
        Q_opt.append(q_opt)
        Qi_KL.append(compute_Qi_KL(Q_true[j][-m_test:],Q_pred[j][-m_test:]))
        F1_opt.append(F1_score(Q_true[j][-m_test:],Q_pred[j][-m_test:],a_opt,b_opt))
        MSE.append(np.mean((Q_true[j][-m_test:] - Q_pred[j][-m_test:])**2))
        MAE.append(np.mean(np.abs(Q_true[j][-m_test:] - Q_pred[j][-m_test:])))

        for i in range(len(EE_rates)):

            # For fixed EE rate, find threshold that maximizes F1 score on training and validation data
            a = Q_true[j][:m_train_val]
            b = Q_pred[j][:m_train_val]
            a_hat, B_hat, F1_scores = F1_vals(a,b,EE_rates[i])
            b_hat = B_hat[np.argmax(F1_scores)]
            
            # Evaluate test F1 using this threshold
            a = Q_true[j][-m_test:]
            b = Q_pred[j][-m_test:]
            F1[i,j] = F1_score(a,b,a_hat,b_hat)

            # Also find area under precision recall curve for test set
            AUC[i,j] = guth_AUC(a,b,EE_rates[i])

    error_statistics = {'Taus' : Taus,
                        'EE_rates' : EE_rates,
                        'MSE' : MSE,
                        'MAE' : MAE,
                        'alpha_star' : Alpha_star,
                        'Q_opt' : Q_opt,
                        'F1_opt' : F1_opt,
                        'Qi_KL' : Qi_KL,
                        'F1' : F1,
                        'AUC' : AUC}     

    np.save(save_file, error_statistics)

if __name__ == "__main__":
    
    F_p_results_file = '../P_to_q/saved_results/F_p_results.npy'
    F_psi_results_file = '../POD/Psi_to_q/saved_models/F_psi_results.npy'
    F_xi_results_file = '../FFNN/Xi_to_q/saved_models/F_xi_results.npy'
    H_p_results_file = '../P_LSTM_ROM/saved_models/P_ROM_results.npy'
    H_psi_results_file = '../POD/Psi_LSTM_ROM/saved_models/Psi_ROM_results.npy'
    H_xi_results_file = '../FFNN/Xi_LSTM_ROM/saved_models/Xi_ROM_32_results.npy'

    F_p_save_file = './error_statistics/F_p_errors'
    F_psi_save_file = './error_statistics/F_psi_errors'
    F_xi_save_file = './error_statistics/F_xi_errors'
    H_p_save_file = './error_statistics/H_p_errors'
    H_psi_save_file = './error_statistics/H_psi_errors'
    H_xi_save_file = './error_statistics/H_xi_errors'

    compute_and_save_errors(np.load(F_p_results_file, allow_pickle=True).item(), F_p_save_file, smooth=False)
    compute_and_save_errors(np.load(F_psi_results_file, allow_pickle=True).item(), F_psi_save_file, smooth=False)
    compute_and_save_errors(np.load(F_xi_results_file, allow_pickle=True).item(), F_xi_save_file, smooth=False)
    compute_and_save_errors(np.load(H_p_results_file, allow_pickle=True).item(), H_p_save_file)
    compute_and_save_errors(np.load(H_psi_results_file, allow_pickle=True).item(), H_psi_save_file)
    compute_and_save_errors(np.load(H_xi_results_file, allow_pickle=True).item(), H_xi_save_file)























