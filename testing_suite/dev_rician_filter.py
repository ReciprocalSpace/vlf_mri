from scipy.stats import rice
from scipy.special import hyp1f1
import numpy as np
import matplotlib.pyplot as plt


def get_signal(t, *args):
    """Get a multi-exponential signal s = sum(A exp(-t/T2)"""
    s = np.zeros_like(t)
    for i in range(0, int(len(args)), 2):
        A = args[i]
        T2 = args[i+1]
        s += A*np.exp(-t/T2)
    return s


def get_noisy_signal(signal, sigma=1., seed=1):
    return rice.rvs(signal/sigma, scale=sigma)


def get_window_sigma(signal, window_size=41):
    std = np.zeros_like(signal, dtype=float)
    for i, _ in enumerate(signal):
        i_min = max(0, i-int(window_size/2+1))
        i_max = min(len(signal), i + int(window_size/2+1))
        std[i] = np.std(signal[i_min:i_max])
    return std

def get_window_moment(signal, window_size=41, order=2):
    moment = np.zeros_like(signal, dtype=float)
    for i, _ in enumerate(signal):
        i_min = max(0, i - int(window_size / 2 + 1))
        i_max = min(len(signal), i + int(window_size / 2 + 1))
        moment[i] = np.mean(signal[i_min:i_max]**order)
    return moment


if __name__ == "__main__":
    t = np.linspace(0., 1., 10001)
    sigma_real = 0.1
    signal = get_signal(t, 0.8, 0.1, 0.2, 0.4)
    noisy_signal = get_noisy_signal(signal, sigma_real, 1)
    mean_noise = rice.mean(signal/sigma_real, scale=sigma_real)

    residue = noisy_signal-mean_noise

    sigma = get_window_sigma(residue, 101)
    sigma_th = rice.std(signal/sigma_real, scale=sigma_real)
    moment2 = get_window_moment(noisy_signal, 101)

    signal_moment = np.sqrt(np.absolute(moment2 - 2 * sigma_real ** 2))

    plt, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].plot(t, noisy_signal)
    ax[0].plot(t, signal_moment)
    ax[0].plot(t, signal)
    ax[2].plot(t, signal_moment-signal)
    # ax[2].plot(t)

    ax[1].plot(t, residue)
    ax[1].plot(t, sigma)
    ax[1].plot(t, sigma_th)
    plt.show()


