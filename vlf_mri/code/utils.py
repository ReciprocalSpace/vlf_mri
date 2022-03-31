from scipy import optimize
import numpy as np


def get_diff_operator(n: int, order: int) -> np.ndarray:
    output = np.eye(n)
    if order == 0:
        return output
    d_f = np.diag(-np.ones(n), k=0) + np.diag(np.ones(n-1), k=1)  # forward derivative
    d_f[-1, -1] = 0
    for _ in range(order):
        output = d_f @ output
    output = output[:(n-order)]
    return output


def ilt_uniform_penalty(t: np.ndarray, signal: np.ndarray, R1: np.ndarray, lmd: float, reg_order=2, penalty="up"):
    M = len(R1)
    N = len(t)
    dR1 = np.gradient(R1)
    dR1 = dR1**0
    laplace_kernel = np.array([np.exp(-t * R) * dR for R, dR in zip(R1, dR1)]).T
    diff_kernel = get_diff_operator(M, order=reg_order)
    Mm = len(diff_kernel)
    bias = np.zeros(( N+Mm, 1))
    bias[:N, 0] = 1

    reg_signal = np.zeros(N+Mm)
    reg_signal[:N] = signal

    alpha_hat = np.zeros(M+1)
    LOSS = []
    for _ in range(5):
        if penalty == "up":
            b_hat = 1 / (1+(diff_kernel @ alpha_hat[:-1])**2)
            p_kernel = np.sqrt(lmd*np.expand_dims(b_hat, axis=1)) * diff_kernel
        elif penalty == "c":
            p_kernel = np.sqrt(lmd) * diff_kernel
        reg_kernel = np.concatenate((laplace_kernel, p_kernel))
        reg_kernel = np.concatenate((reg_kernel, bias), axis=1)

        x, res = optimize.nnls(reg_kernel, reg_signal)
        LOSS.append(res)
    alpha_hat = x[:-1]
    c = x[-1]

    signal_hat = laplace_kernel @ alpha_hat + c

    return alpha_hat, signal_hat, LOSS


def ilt_simple(t: np.ndarray, signal: np.ndarray, R1: np.ndarray, lmd: float):
    M = len(R1)
    N = len(t)
    dR1 = np.gradient(R1)
    dR1 = dR1**0
    laplace_kernel = np.array([np.exp(-t * R) * dR for R, dR in zip(R1, dR1)]).T
    diff_kernel = get_diff_operator(M, order=0)
    Mm = len(diff_kernel)
    bias = np.zeros(( N+Mm, 1))
    bias[:N, 0] = 1

    reg_signal = np.zeros(N+Mm)
    reg_signal[:N] = signal

    alpha_hat = np.zeros(M+1)
    LOSS = []
    for _ in range(5):
        b_hat = 1 / (1 + (diff_kernel @ alpha_hat[:-1]) ** 2)
        up_kernel = np.sqrt(lmd * np.expand_dims(b_hat, axis=1)) * diff_kernel

        reg_kernel = np.concatenate((laplace_kernel, up_kernel))
        reg_kernel = np.concatenate((reg_kernel, bias), axis=1)

        x, res = optimize.nnls(reg_kernel, reg_signal)
        LOSS.append(res)
    alpha_hat = x[:-1]
    c = x[-1]

    signal_hat = laplace_kernel @ alpha_hat + c

    return alpha_hat, signal_hat, LOSS