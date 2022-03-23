import numpy as np
import numba as nb

@nb.njit
def precision(a,b,a_hat,b_hat):
    """
    Precision of classifier (b >= b_hat) for truth (a >= a_hat)
    """

    a_class = np.zeros_like(a)
    b_class = np.zeros_like(b)

    for j in range(len(a)):

        if a[j] >= a_hat:
            a_class[j] = 1

        if b[j] >= b_hat:
            b_class[j] = 1
    
    TP = np.sum(a_class*b_class)
    FP = np.sum((1-a_class)*b_class)
    
    if TP+FP==0: 
        s = 0
    else:
        s = TP/(TP+FP)
    
    return s

@nb.njit
def recall(a,b,a_hat,b_hat):
    """
    Recall of classifier (b >= b_hat) for truth (a >= a_hat)
    """
    
    a_class = np.zeros_like(a)
    b_class = np.zeros_like(b)

    for j in range(len(a)):

        if a[j] >= a_hat:
            a_class[j] = 1

        if b[j] >= b_hat:
            b_class[j] = 1
    
    TP = np.sum(a_class*b_class)
    FN = np.sum(a_class*(1-b_class))
    
    if TP+FN==0: 
        r = 1
    else:
        r = TP/(TP+FN)
    
    return r

def F1_score(a,b,a_hat,b_hat):
    """
    F1 score of classifier (b >= b_hat) for truth (a >= a_hat)
    """
    s = precision(a,b,a_hat,b_hat)
    r = recall(a,b,a_hat,b_hat)

    if r+s == 0:
        # i.e. no true positives
        return 0

    else:
        F1 = 2*(r*s)/(r+s)
        return F1

#@nb.njit   
def grad_recall(a,b,a_hat,b_hat,eps):
    """
    returns gradient of recall with respect to classifier threshold, dr/db
    """
    
    if b_hat - eps < np.min(b):
        recall_b_plus = recall(a,b,a_hat,b_hat+eps)
        recall_b = recall(a,b,a_hat,b_hat)
        drdb = (recall_b_plus - recall_b) / eps
        
    elif b_hat + eps > np.max(b):
        recall_b_minus = recall(a,b,a_hat,b_hat-eps)
        recall_b = recall(a,b,a_hat,b_hat)
        drdb = (recall_b - recall_b_minus) / eps
        
    else:
        recall_b_minus = recall(a,b,a_hat,b_hat-eps)
        recall_b_plus = recall(a,b,a_hat,b_hat+eps)
        drdb = (recall_b_plus - recall_b_minus) / (2*eps)
    
    return drdb

def guth_AUC(a,b,q,nb=501,eps=None):
    """
    Computes the integral in equation 11 in Guth and Sapsis, Entropy, 2019
    """
    
    if eps is None: eps = 1/(nb-1)

    # Get a_hat from q
    a_hat = np.percentile(a, 100*(1-q))

    # Range of meaningful values of b_hat
    B_hat = np.linspace(np.min(b),np.max(b),nb)
    
    # Precision and gradient of recall w.r.t. b_hat
    Precision = np.zeros_like(B_hat)
    Recall = np.zeros_like(B_hat)
    Grad_recall = np.zeros_like(B_hat)

    for j in range(nb):
        Precision[j] = precision(a,b,a_hat,B_hat[j])
        Recall[j] = recall(a,b,a_hat,B_hat[j])
    
    Grad_recall[0] = (Recall[1]-Recall[0])/(B_hat[1] - B_hat[0])
    Grad_recall[-1] = (Recall[-1]-Recall[-2])/(B_hat[1] - B_hat[0])
    Grad_recall[1:-1] = (Recall[2:]-Recall[:-2])/(B_hat[2] - B_hat[0])

    # AUC = \int_{min(b)}^{max(b)} s(a_hat, b_hat) * |d/db_hat r(a_hat,b_hat)| db_hat
    PgR = Precision * np.abs(Grad_recall)

    if nb % 2 == 1:
        # Simpson's rule
        AUC = (B_hat[1]-B_hat[0])/3 * np.sum(PgR[0:-1:2] + 4*PgR[1::2] + PgR[2::2])
    else:
        # Trapezoidal integral 
        AUC = np.sum(PgR[1:] + PgR[:-1]) * (B_hat[1]-B_hat[0])/2

    return AUC

def guth_criterion(a,b,nq=50,nb=101,q_min=0,q_max=0.2,return_thresholds=False,Q=None):
    """
    Computes the criterion descibed by equation 17 in Guth and Sapsis, Entropy, 2019

    Max is taken over a grid of nq values of q (extreme event rate) from 1/nb to q_max
    Inputs:
        a   : dataset with extreme events (required)
        b   : indicator of extreme events in a (required)
        nq  : number of extreme events rates to search
        nb  : number of discretization points for integral in Eq. 11
        q_max : max extreme event rate to use in search
        return_threshold : if yes, then return q that maximizes Eq. 17, corresponding a_hat, and 
                           b_hat that maximizes F1 score
        Q   : option to pass in values of q to search
    Returns:
        alpha_star : see Eq. 17
        q_opt      : q that maximizes Eq. 17
        a_opt      : corresponding threshold for a
        b_opt      : threshold for b that maximizes F1 score
    """
    
    if Q is None: Q = np.linspace(q_min,q_max,nq+1)[1:]
    gAUC = np.array([guth_AUC(a,b,q) for q in Q])
    alpha_q = gAUC-Q
    alpha_star = np.max(alpha_q)
    
    if return_thresholds:

        # EE rate yielding best criterion
        q_opt = Q[np.argmax(alpha_q)]

        # Corresponding a_hat
        a_opt = np.percentile(a, 100*(1-q_opt))

        # b_hat that maximizes F1 score, given a_hat = a_opt
        B_hat = np.linspace(np.min(b),np.max(b),nb)
        b_opt = B_hat[np.argmax([F1_score(a,b,a_opt,b_hat) for b_hat in B_hat])]

        return alpha_star, q_opt, a_opt, b_opt

    else:
        return alpha_star

def F1_vals(a,b,q=0.1,nb=101):
    """
    Computes F1 scores for a range of thresholds on predicted data given desired extreme event rate.

    Inputs:
        a   : dataset with extreme events (required)
        b   : indicator of extreme events in a (required)
        q   : extreme event rate for a
        nb  : number of thresholds to check for b
    Returns:
        a_hat : threshold corresponding to extreme event rate q 
        B_hat : vector of thresholds used to compute F1 scores
        F1_scores : F1 scores using thresholds a_hat and each b_hat in B_hat
    """

    # Extreme event cutoff for a
    a_hat = np.percentile(a, 100*(1-q))

    # Cutoffs to check for b
    B_hat = np.linspace(np.min(b),np.max(b),nb)

    # F1 scores for each b_hat in B_hat
    F1_scores = [F1_score(a,b,a_hat,b_hat) for b_hat in B_hat]

    return a_hat, B_hat, F1_scores