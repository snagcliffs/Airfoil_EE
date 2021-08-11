import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import filters

def save_P(min_time):

    pres_hist = np.load('pres_hist.npy')
    time_hist = np.load('time_hist.npy')
    min_ind = np.min(np.where(time_hist > min_time))

    pres_hist = (pres_hist - np.mean(pres_hist[min_ind:,:], axis=0)) / np.std(pres_hist[min_ind:,:], axis=0)

    P = np.hstack([time_hist.reshape(len(time_hist),1), pres_hist])
    print(P.shape)
    np.save('P',P)

def save_q(min_time):

    # Load force coefficients
    forceCoeffs = np.load('./forceCoeffs.npy')
    sim_time = forceCoeffs[100:,0]
    Cd = forceCoeffs[100:,1]
    Cl = forceCoeffs[100:,2]
    dt = sim_time[1]-sim_time[0]
    min_ind = np.min(np.where(sim_time > min_time))

    # Get peak frequency of drag coefficient and set up smoother
    F_Cd = np.fft.fft(Cd[min_ind:] - np.mean(Cd[min_ind:]))
    freqs = np.fft.fftfreq(len(Cd[min_ind:]), d=dt)
    f_peak = freqs[np.argmax(np.abs(F_Cd))]

    width_smoother = int(3/(f_peak*dt))
    scale_smoother = int(0.5/(f_peak*dt))
    smoother_kern = gaussian(width_smoother, scale_smoother)
    smoother_kern = smoother_kern/np.sum(smoother_kern)

    # Quantity of interest as smoothed quotient Cd/Cl
    q = filters.convolve1d(Cd, smoother_kern)
    q = (q - np.mean(q[min_ind:]))/np.std(q[min_ind:])

    q = np.stack([sim_time, q]).T
    print(q.shape)
    np.save('q', q)

def main(min_time = 20):
    save_q(min_time)
    save_P(min_time)

if __name__ == "__main__":

    main()

