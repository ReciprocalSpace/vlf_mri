from scipy import optimize
import numpy as np


def get_diff_operator(n: int, order: int) -> np.ndarray:
    """Return the i th order derivation operator for a vector of length n

    Parameters
    ----------
    n: int
        Length of the independent variable vector. I.e. if y = f(x), n = len(x).
    order: int
        Order of the differentiation operator to obtain
    Returns
    -------
    diff_operator: np.ndarray
        2-dimensional rectangular matrix corresponding to the discrete differentiation operator
    """
    output = np.eye(n)
    if order == 0:
        return output
    d_f = np.diag(-np.ones(n), k=0) + np.diag(np.ones(n-1), k=1)  # forward derivative
    d_f[-1, -1] = 0
    for _ in range(order):
        output = d_f @ output
    output = output[:(n-order)]
    return output


def inverse_laplace_transform(t: np.ndarray, signal: np.ndarray, R1: np.ndarray, lmd: float, reg_order=2, penalty="up",
                              boundary_clamp: int = 5):
    """Performs the inverse laplace transform of a signal

    Parameters
    ----------
    t: numpy.ndarray
        Independent variable (time)
    signal: numpy.ndarray
        Dependant variable (magnetization)
    R1: numpy.ndarray
        Array of R1 values for which the ILT will be performed.
    lmd: float
        Regularization parameters used to perform the ILT
    reg_order: int, default 2
        Order the differential operator used for regularization
    penalty: str, 'up' or 'c', default is 'up'
        Penalty used for the regularization. 'up': uniform penalty, and 'c' for constant. The uniform penalty tries
    boundary_clamp: int, default 5
        Boundary conditions, define the number of points "clamped" to zero on both sides of the R1 domain.

    Returns
    -------
    alpha_hat: numpy.ndarray
        Density for each R1 values (populations)
    signal_hat: numpy.ndarray
        Signal approximation using the Laplace Transform of the density alpha_hat
    loss: float
        Mean root square of the error : 1/n*(sum((signal-signal_hat)**2))**0.5
    """

    # The operator if built piece by piece and contains three parts:
    # [ A c ]
    # [ B c ]
    # A: the laplace kernel -> this is the actual operator that we need to inverse to compute the ILF
    # B: The regularization kernel -> this part constrains the solution to a space of smooth and "desirable" solutions
    # c: a bias kernel -> an additional constant parameter in the model

    M = len(R1)
    N = len(t)

    # A "true" inverse laplace transform would require dR parameter -> the integral is performed over a non-uniform
    # domain, but this does not work very well with the regularization kernel.

    # dR1 = np.gradient(R1)
    # laplace_kernel = np.array([np.exp(-t * R) * dR for R, dR in zip(R1, dR1)]).T

    laplace_kernel = np.array([np.exp(-t * R) for R in R1]).T
    diff_kernel = get_diff_operator(M, order=reg_order)

    Mm = len(diff_kernel)
    bias = np.zeros(( N+Mm, 1))
    bias[:N, 0] = 1

    # The signal must be extended with zeros to match the shape of the operator
    reg_signal = np.zeros(N + Mm)
    reg_signal[:N] = signal

    alpha_hat = np.zeros(M+1)  # M+1 because of the constant

    # If the chosen penalty is "up", the ILT must be solved iteratively using a while loop. If the chosen penalty is not
    # "up", then the loop is terminated at the end of the first iteration
    loss_m1 = 1e308
    j = 0
    while True:
        if penalty == "up":
            b_hat = 1 / (1+(diff_kernel @ alpha_hat[:-1])**2)
            p_kernel = np.sqrt(lmd*np.expand_dims(b_hat, axis=1)) * diff_kernel

        elif penalty == "c":
            p_kernel = np.sqrt(lmd) * diff_kernel

        else:
            raise NotImplementedError

        # boundary condition -> puts a large number on the first and last n diagonal element of the matrix, clamping
        # the first and last values of the alpha_hat solution vectors to 0.
        large_number = np.max(p_kernel) * 1e6
        for i in range(boundary_clamp):
            p_kernel[i, i] = large_number
            p_kernel[-1-i, -1-i] = large_number

        # Here we assemble the three kernels
        reg_kernel = np.concatenate((laplace_kernel, p_kernel))
        reg_kernel = np.concatenate((reg_kernel, bias), axis=1)

        # Kernel is solved using a non-negative least square algorithm
        alpha_hat, loss = optimize.nnls(reg_kernel, reg_signal)

        if penalty == "c":
            break
        elif abs(loss_m1-loss) < 1e-8:
            break
        elif j > 10:
            print("ILT: did not converge")
            break

        j += 1
        loss_m1 = loss

    c = alpha_hat[-1]
    alpha_hat = alpha_hat[:-1]
    signal_hat = laplace_kernel @ alpha_hat + c

    return alpha_hat, signal_hat, loss
