from typing import Union, Tuple

import copy
from collections.abc import Iterable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patheffects
from cycler import cycler
from lmfit import Model, Parameters

import vlf_mri
from .vlf_data import VlfData
from .rel_data import RelData


class ILTData(VlfData):
    """Container for inverse Laplace transform data

    Attributes
    ----------
    B_relax: numpy.ndarray
        Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID
        data and to a 1D array of tau.
    R1: numpy.ndarray
        Array of R1 values for which the ILT was performed.
    lmd: float
        Regularization parameters used to perform the ILT
    """

    def __init__(self, fid_file_path: Path, algorithm: str, alpha: np.ndarray, B_relax: np.ndarray,
                 R1: np.ndarray, lmd: np.ndarray, mask=None, best_fit=None, normalize=True):
        """

        Parameters
        ----------
        fid_file_path: Path
            Path to the original *.sdf file containing the experimental data
        algorithm: str
            Name of the algorithm used to produces the ILT
        alpha: numpy.ndarray
            Array of data, correspond to the populations (density) of the ILT
        B_relax: numpy.ndarray
            Array of explored relaxation magnetic fields. Each element of this array corresponds to a 1D array of ILT.
        R1: numpy.ndarray
            Array of R1 values for which the ILT was performed.
        lmd: float
            Regularization parameters used to perform the ILT
        mask : numpy.ndarray of bool, optional
            Mask to apply on data array. Must be the same dimension as data array. Default is an array of False.
        best_fit : dict, optional
            Dictionary containing fitted model of the data. For each element of best_fit, the key is the name of the
            algorithm, while the value is a FitData or a FitDataArray object. An entry is added to the dictionary every
            time an algorithm is run on the data. Default is an empty dictionary.
        normalize : bool, optional
            Signal if the ilt_data must be normalized or not. Default is True.
        """
        super().__init__(fid_file_path, "ILT", alpha, mask, best_fit)

        if normalize:
            self.data /= np.sum(self.data, axis=1, keepdims=True)  # Normalization

        self.B_relax = B_relax
        self.R1 = R1
        self.lmd = lmd

    def batch_plot(self, title=None, save=""):
        # Prepare the figure and axes
        plt.figure(figsize=(8 / 2.54, 10 / 2.54), tight_layout=True, dpi=150)
        ax = plt.gca()
        colormap = plt.cm.get_cmap("plasma")
        custom_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.B_relax))])
        ax.set_prop_cycle(cycler('color', custom_colors))

        # Plot each line
        cst = np.zeros_like(self.R1[0])
        offset = np.linspace(0, 1, len(self.R1))
        for i, (R1_i, alpha_i, B_relax_i, lmd_i, offset_i) in \
                enumerate(zip(self.R1, self.data, self.B_relax, self.lmd, offset)):

            plt.semilogx(R1_i, cst + offset_i, c="k", alpha=0.2, linewidth=1)
            plt.semilogx(R1_i, alpha_i + offset_i)
            plt.title(title)
            plt.xlabel(r"Relaxation $R_1$ [s$^{-1}$]")
            plt.ylabel(r"Density $\alpha$ [arb. unit]")

            if i % (len(self.R1) // 15 + 1) == 0:
                txt = plt.text(x=R1_i.min(), y=offset_i + 0.01, s=rf"{B_relax_i:.1e}", fontsize=6)
                txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w', alpha=0.75)])

        plt.ylim((-0.025, offset[-1] + 0.1))
        plt.text(x=self.R1[0].min(), y=offset[-1] + 0.04, s=r"$B_{rel}$", fontsize=8)
        plt.text(x=self.R1[0].max(), y=offset[-1] + 0.04, s=r"$\lambda$=" + f"{self.lmd[0]:.1e}", fontsize=6,
                 horizontalalignment='right', alpha=0.75)
        if save:
            plt.savefig(save)
        plt.show()

    def __getitem__(self, item):
        # Defines the slices objects used to slice through the objects
        sl = []
        if not isinstance(item, Iterable):  # 2D or 3D slices
            item = (item,)
        for it in item:
            sl.append(it if isinstance(it, slice) else slice(it, it + 1))
        while len(sl) < 2:
            sl.append(slice(0, None))

        sl = tuple(sl)

        data_file_path = self.data_file_path
        alpha = self.data[sl]
        B_relax = self.B_relax[sl[0]]
        R1 = self.R1[sl[0:2]]
        lmd = self.lmd
        mask = self.mask[sl]
        best_fit = {key: value[sl] for key, value in self.best_fit.items()}

        # return ILTData(data_file_path, fid_matrix, B_relax, tau, t_fid, mask, best_fit)
        return ILTData(data_file_path, '', alpha, B_relax, R1, lmd, mask, best_fit, False)

    def _get_r11(self, order: int, display: True) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Extract dispersions R1(B0) from ILTs

        This private method extracts n dispersions from the ilt_data object. It works by fitting the ilt with a series
        of gaussians applied sequentially. The algorithm goes as follows:
        - For i=1,...,order
            - For each ILT
                - signal_fitted <- gaussian fit of signal (global maxima)
                - signal <- signal - fitted_signal
                - signal <- signal - fitting_artifact (ripples in the correction)
                - get mu parameter of the fitter gaussian
            - dispersion <- list of mu parameters
        - returns list of dispersions

        This algorithm might not be very robust. To use with caution!

        Parameters
        ----------
        order: int
            Number of dispersion to extract

        Returns
        -------
        dispersions: numpy.ndarray or tuple of numpy.ndarray
            N dispersions
        """

        def my_fun(R1, mu, sigma, A):
            return A * np.exp(-(R1 - mu) ** 2 / sigma ** 2)

        new = copy.deepcopy(self)

        result = []
        for j in range(order):
            dispersion = []
            population = []

            if display:
                new.batch_plot(f"Gaussian fit # {j}")

            for R1_i, alpha_i in zip(new.R1, new.data):
                # Initializing model
                mod = Model(my_fun, independent_vars=["R1"])
                mu0 = R1_i[np.argmax(alpha_i)]  # Initial guess

                params = Parameters()
                params.add('A', value=alpha_i.max(), min=0.)
                params.add('mu', value=np.log(mu0), min=0)
                params.add('sigma', value=1, min=0)

                res = mod.fit(alpha_i, params, R1=np.log(R1_i))

                # Saving results
                dispersion.append(np.exp(res.params["mu"].value))
                population.append(res.params["A"].value)

                # Preparing for next iteration
                # err = np.clip(alpha_i - res.best_fit, 0, None)
                err = np.absolute(alpha_i - res.best_fit)  # clip works better, but might lead to unforeseen bugs !

                alpha_i[:] = err - np.clip(err, None, res.params["A"].value / 8) * \
                             np.power(res.best_fit / res.best_fit.max(), 0.15)
            result.append((np.array(dispersion),
                           np.array(population)))

        if display:
            new.batch_plot(f"Gaussian fit # {j+1}")

        result = result[0] if len(result) == 1 else tuple(result)

        return result

    def to_rel(self, display=False):
        # TODO: add a fitted version of the data to the best_fit attribute

        ((disp1, pop1), (disp2, pop2)) = self._get_r11(2, display)

        alpha = pop1 / (pop1 + pop2)
        data = np.array([disp1, disp1, disp2, alpha])

        return vlf_mri.RelData(self.data_file_path, data, self.B_relax)

    def report(self):
        # TODO: produce a standardised report for the ILTData class
        raise NotImplemented








