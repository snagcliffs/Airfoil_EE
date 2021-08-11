import numpy as np
from tqdm import tqdm

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def compute_Xi(model, datapath, savepath, stride):

    # Load pressure from file
    # Skip first line since it has a shorter timestep.  Data is otherwise uniformly spaced.
    S = int(100 / model.n) * np.arange(model.n)
    PT = np.load(datapath + 'P.npy')
    P_time = PT[:,0]
    P = PT[:,1:][:,S]

    m_hist = model.m_hist
    m = len(P_time) - (m_hist-1)*stride
    batch_size = 2000
    n_batches = int(np.ceil(m / batch_size))

    Xi = []
    Xi_time = []

    # Loop over inputs in batches
    for batch in tqdm(range(n_batches)):

        # Get pressure sensor history for current batch
        P_hist_batch = []
        t_batch = []

        for j in range(batch_size):

            max_ind = (m_hist-1)*stride+j+batch_size*batch
            if max_ind >= len(P_time): break
            inds = [max_ind-i*stride for i in range(m_hist)]
            P_hist_batch.append(P[inds[::-1],:])
            t_batch.append(P_time[max_ind])

        P_hist_batch = np.array(P_hist_batch)
        Xi_time.append(np.array(t_batch))

        # Get low dimensional encoding
        Xi.append(model.encode(P_hist_batch))

    Xi = np.concatenate(Xi, axis=0)
    Xi_time = np.concatenate(Xi_time, axis=0)

    # Load corresponding q
    tq = np.load(datapath + 'q.npy')
    fc_time = tq[:,0]
    q = tq[:,1]
    fc_snapshot_inds = np.argmin(np.abs(fc_time - Xi_time[0]))+10*np.arange(m)
    q_course = q[fc_snapshot_inds]

    Xi_dict = {'Xi' : Xi, 'time' : Xi_time, 'q' : q_course}
    np.save(savepath + 'Xi_' + str(model.r), Xi_dict)






