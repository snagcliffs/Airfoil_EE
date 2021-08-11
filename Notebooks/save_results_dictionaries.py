import numpy as np
from tqdm import tqdm

def save_results_dict(base_path, save_file):

	Results = {}
	Taus = [np.round(0.7*j,1) for j in range(16)]

	for j in tqdm(range(len(Taus))):
	    tau = Taus[j]
	    Results[tau] = np.load(base_path+'results_tau'+str(tau)+'.npy',allow_pickle=True).item()

	np.save(base_path + save_file, Results)

if __name__ == "__main__":
    
    F_p_results_path = '../P_to_q/saved_results/'
    F_psi_results_path = '../POD/Psi_to_q/saved_models/'
    F_xi_results_path = '../FFNN/Xi_to_q/saved_models/'

    F_p_save_file = 'F_p_results'
    F_psi_save_file = 'F_psi_results'
    F_xi_save_file = 'F_xi_results'

    save_results_dict(F_p_results_path, F_p_save_file)
    save_results_dict(F_psi_results_path, F_psi_save_file)
    save_results_dict(F_xi_results_path, F_xi_save_file)



