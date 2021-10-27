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

from vlf_mri.lib.pdf_saver import PDFSaver
from vlf_mri.lib.vlf_data import VlfData


class MagData(VlfData):
    def __init__(self, fid_file_path, algorithm, mag_matrix, B_relax, tau, mask=None, best_fit=None):
        super().__init__(fid_file_path, "MAG", best_fit)
        self.algorithm = algorithm
        self.mag_matrix = mag_matrix
        self.B_relax = B_relax
        self.tau = tau

        self.mask = np.zeros_like(mag_matrix) if mask is None else mask

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
