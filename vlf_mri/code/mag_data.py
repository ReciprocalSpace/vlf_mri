# from vlf_mri.lib.data_reader import import_SDF_file
# import logging
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC
from collections.abc import Iterable
from cycler import cycler
from lmfit import Model, Parameters, report_fit
from math import ceil
from numpy import ma
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rice
from tqdm import tqdm

from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData


class MagData(VlfData):
    def __init__(self, fid_file_path, algorithm, mag_matrix, B_relax, tau, mask=None, best_fit=None):
        if best_fit is None:
            best_fit = {}
        super().__init__(fid_file_path, "MAG", best_fit)

        self.algorithm = algorithm
        self.mag_matrix = mag_matrix
        self.B_relax = B_relax
        self.tau = tau
        self.mask = np.zeros_like(mag_matrix, dtype=bool) if mask is None else mask

        self._normalize()
        self.mask = np.logical_or(self.mask, self.mag_matrix <= 0)

    def _normalize(self):
        output = []

        for mag_matrix_i, mask_i in zip(self.mag_matrix, self.mask):
            unmask_data = mag_matrix_i[~mask_i]
            cond = unmask_data[0] < unmask_data[-1]
            m0 = unmask_data.min() if cond else unmask_data.max()
            m1 = unmask_data.min() if not cond else unmask_data.max()
            output.append((m0 - mag_matrix_i) / (m0 - m1))

        output = np.array(output)
        output[output == 0.] = 1e-12  # Avoid errors with max_likelihood approaches

        self.mag_matrix = output

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
        algorithm = self.algorithm
        mag_matrix = self.mag_matrix[sl]
        B_relax = self.B_relax[sl[0]]
        tau = self.tau[sl]
        mask = self.mask[sl]
        best_fit = {key: value[sl] for key, value in self.best_fit.items()}

        return MagData(data_file_path, algorithm, mag_matrix, B_relax, tau, mask, best_fit)

    def apply_mask(self, sigma=2., dims="xy", display_report=False) -> None:
        """ Mask aberrant values in mag_matrix

        Find and mask aberrant values in the fid data matrix. Aberrant values are defined as either negative values
        or discontinuities in the signal. Discontinuities in the data are found by evaluating the normalized gradient
        of the signal toward all dimensions, and rejecting the data points whose normalized gradient magnitude is
        greater than a chosen sigma.

        The normalized gradient towards the i^th  dimension is defined as (grad_i(FID) - mean(grad_i(FID))/
        std(grad_i(FID).

        If display_masked_data is True, then the data result of

        Parameters
        ----------
        sigma: float
            Rejection criterion for the data points. Default is 2.

        display_report: bool
            Display the results of masking procedure.

        Returns
        -------
        None
        :param dims:
        """
        # Apply a mask aberrant data
        ind = self.mag_matrix <= 0.
        self.mask[ind] = True

        grad = np.array(np.gradient(self.mag_matrix))
        # Normalize gradient array
        axis = (1, 2)
        grad_mean = np.expand_dims(grad.mean(axis=axis), axis=axis)
        grad_std = np.expand_dims(grad.std(axis=axis), axis=axis)
        grad = (grad - grad_mean) / grad_std

        # grad_n = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        grad_B = grad[0] if 'x' in dims else 0
        grad_tau = grad[1] if 'y' in dims else 0
        grad_n = np.sqrt(grad_B ** 2 + grad_tau ** 2)

        self.mask = np.logical_or(self.mask, grad_n > sigma)

        if display_report:
            mag_matrix = ma.masked_array(self.mag_matrix, mask=self.mask)
            fig, (axes_bef, axes_after) = plt.subplots(2, 2, tight_layout=True, figsize=(15 / 2.54, 10 / 2.54))

            fig.suptitle("Report: Apply mask")
            grad_B, grad_tau = tuple(grad.reshape((2, -1)))
            grad_n = grad_n.reshape(-1)
            ind = self.mask.reshape(-1)
            nb_pts = len(grad_n)

            # First row
            # First ax
            axes_bef[0].plot(grad_B[~ind], grad_tau[~ind], ".", c="b",
                             label=f"Good: {np.sum(~ind) / nb_pts * 100:.2f}%")
            axes_bef[0].plot(grad_B[ind], grad_tau[ind], ".", c="r", label=f"Bad:  {np.sum(ind) / nb_pts * 100:.2f}%")
            axes_bef[0].set_xlabel(r'Gradient $\nabla_B$(FID)', labelpad=0)
            axes_bef[0].set_ylabel("BEFORE\nGradient " r"$\nabla_\tau$(FID)", labelpad=0)
            axes_bef[1].hist(np.log(grad_n + 1), 30, density=True)  # +1 to avoid log(0)
            axes_bef[1].set_xlabel(r"Norm $\log(\|\nabla\|)$")
            axes_bef[1].set_ylabel(r"Point density")
            # Second row
            # First ax
            axes_after[0].plot(grad_B[~ind], grad_tau[~ind], ".", c="b", label="_")
            axes_after[0].set_xlabel(r'Gradient $\nabla_B$(FID)', labelpad=0)
            axes_after[0].set_ylabel("AFTER\nGradient " r"$\nabla_\tau$(FID)", labelpad=0)
            # Second ax
            axes_after[1].hist(np.log(grad_n[~ind] + 1), 30, density=True)  # +1 to avoid log(0)
            axes_after[1].set_xlabel(r"Norm $\log(\|\nabla\|)$", labelpad=0)
            axes_after[1].set_ylabel("Point density", labelpad=0)
            # Finishing legend and labels...
            lines_labels = [ax.get_legend_handles_labels() for ax in np.array(fig.axes).flatten()]
            for ax in np.array(fig.axes).flatten():
                ax.tick_params(axis='both', which='major', pad=0)
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title=f"{nb_pts} data points")

            plt.show()

            self.batch_plot("Report: MAG after mask")


    @staticmethod
    def model_mono_exp(tau, amp, R1):
        return amp * np.exp(-R1 * tau)

    @staticmethod
    def model_bi_exp(tau, amp, alpha, R11, R12):
        return amp * alpha * np.exp(-R11 * tau) + amp * (1 - alpha) * np.exp(-R12 * tau)

    def adjust_mono_exp(self, tau, magnetization):
        mod = Model(self.model_mono_exp, independent_vars=['tau'])
        params = Parameters()
        R1 = 1 / tau[np.argmin(np.absolute(magnetization - 0.63))]
        params.add('amp', value=1., min=0.)
        params.add('R1', value=R1, min=0)
        ind = ~magnetization.mask
        result_mono = mod.fit(magnetization[ind], params, tau=tau[ind])
        result_mono.best_fit = self.model_mono_exp(tau,
                                              result_mono.params['amp'].value,
                                              result_mono.params['R1'].value)
        return result_mono

    def to_rel_biexp(self, tau, magnetization):
        mod = Model(self.model_bi_exp, independent_vars=['tau'])
        params = Parameters()
        R1 = 1 / tau[np.argmin(np.absolute(magnetization - 0.63))]
        params.add('amp', value=1., min=0.8, max=1.2)
        params.add('alpha', value=0.5, min=0., max=1.)
        params.add('R11', value=R1, min=0, max=100)
        params.add('R12', value=R1 * 2, min=0, max=100)
        ind = ~magnetization.mask
        result_bi = mod.fit(magnetization[ind], params, tau=tau[ind])
        # report_fit(result_bi)
        result_bi.best_fit = self.model_bi_exp(tau,
                                          result_bi.params['amp'].value,
                                          result_bi.params['alpha'].value,
                                          result_bi.params['R11'].value,
                                          result_bi.params['R12'].value)
        return result_bi

    def get_relaxation_times(self, tau, list_of_magnetization, save_folder="", manip_name=""):
        result_mono = []
        result_bi = []
        for tau_i, magnetization in zip(tau, list_of_magnetization):
            result_mono.append(self.adjust_mono_exp(tau_i, magnetization))
            result_bi.append(self.adjust_bi_exp(tau_i, magnetization))

        if save_folder != "" and manip_name != "":
            pass
        return result_mono, result_bi

    def save_to_pdf(self, display):
        file_name = f"{self.experience_name}_Aimantation.pdf"
        file_path = self.saving_folder / file_name
        title = f"{self.experience_name} - Magnetization"

        pdf = PDFSaver(file_path, 2, 4, title, True)
        for tau_i, mag_i, B_relax_i, mono_exp_i, bi_exp_i in zip(self.tau, self.mag_matrix, self.B_relax,
                                                                 self.best_fit['mono_exp'], self.best_fit["bi_exp"]):
            ax = pdf.get_ax()
            ax.set_xlabel('$tau$ u.a.', fontsize=8)
            ax.plot(tau_i, mag_i, '*', markersize=5,
                    label=r"$B_{relax}$" + f" = {B_relax_i:.2e} MHz")
            ax.plot(tau_i, mono_exp_i.best_fit, '--', c='tab:pink', lw=3,
                    label=r"$R_{1}$" + f"= {mono_exp_i.params['R1'].value:.2f}")
            ax.plot(tau_i, bi_exp_i.best_fit, '--', c='tab:olive', lw=3,
                    label=(r"$R_1^{(1)}$" + f"={bi_exp_i.params['R11'].value:.2f}\n"
                                            r"$R_1^{(2)}$" + f"={bi_exp_i.params['R12'].value:.2f}"))
            ax.legend(loc="lower left", fontsize='xx-small', handlelength=1, handletextpad=0.2)
            ax.set_xscale('log')
            ax.grid()

            ax = pdf.get_ax()
            ax.set_xlabel('$tau$ u.a.', fontsize=8)
            ax.set_ylabel('Residu', fontsize=8)
            ax.plot(tau_i, mag_i - mono_exp_i.best_fit, '*', c='tab:pink', markersize=4,
                    label=r"$R_{1}$" + f"= {mono_exp_i.params['R1'].value:.2f}")
            ax.plot(tau_i, mag_i - bi_exp_i.best_fit, '*', c='tab:olive', markersize=4,
                    label=(r"$R_1^{(1)}$" + f"={bi_exp_i.params['R11'].value:.2f}\n"
                                            r"$R_1^{(2)}$" + f"={bi_exp_i.params['R12'].value:.2f}"))
            ax.set_xscale('log')
            ax.grid()
        pdf.close_pdf()

    def batch_plot(self, suptitle="") -> None:
        mag_data = ma.masked_array(self.mag_matrix, self.mask)
        n_fig = len(self.B_relax)
        n_rows, n_columns = ceil(n_fig / 3), 3
        fig, axes_2D = plt.subplots(n_rows, n_columns, sharey="all",
                                    figsize=(16 / 2.54, n_rows * 5 / 2.54),
                                    squeeze=False)
        axes_1D = np.array(axes_2D).flatten()

        colormap = plt.cm.get_cmap("plasma")
        costum_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.tau.T))])

        # costum_cycler = cycler(color=[colormap(x) for x in np.linspace(0, 0.8, len(tau))])
        for i, ax in enumerate(axes_1D):
            ax.set_prop_cycle(cycler('color', costum_colors))
            if i < len(mag_data):
                cst = np.max(np.absolute(mag_data[i]))
                ax.plot(self.tau[i], mag_data[i] / cst, "--.")
                ax.set_xscale("log")
                ax.tick_params(axis='both', which='major', pad=0)
                ax.text(0, 0.9, r"$B_{relax}$=" f"{self.B_relax[i]:.1e} MHz")
            else:
                ax.axis('off')
        for axes_i in axes_2D:
            axes_i[0].set_ylabel("Mag. [arb. units]")

        for ax in axes_2D[-1]:
            ax.set_xlabel(r"Time $\tau$ [us]")
        fig.suptitle(suptitle)
        plt.subplots_adjust(wspace=0)
        plt.show()
