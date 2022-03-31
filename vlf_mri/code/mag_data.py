import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from cycler import cycler
from lmfit import Model, Parameters
from math import ceil
from numpy import ma
from pathlib import Path


from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData
from vlf_mri.code.rel_data import RelData
from vlf_mri.code.fit_data import FitDataArray, FitData
from vlf_mri.code.ilt_data import ILTData
from . import utils


class MagData(VlfData):
    """
        Contain and manage magnetization data

        This class stores and manages magnetization data.

        Attributes
        ----------
        B_relax : numpy.ndarray
            Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID
            data and to a 1D array of tau.
        tau : numpy.ndarray
            Array of explored incrementation times. Each element of this array corresponds to a 1D array of FID data.
        algorithm : str
            Algorithm used to extract the magnetization values

        Methods
        -------
        apply_mask(sigma, dims, display_report):
            Mask aberrant values in data
        batch_plot(title):
            Plot all FID data in a single figure
        save_to_pdf(fit_to_plot, display):
            Produce a pdf with the magnetization data
        to_rel():
            Extract the relaxation time and populations from the magnetization data
        """
    def __init__(self, fid_file_path: Path, algorithm: str, mag_data: np.ndarray, B_relax: np.ndarray,
                 tau: np.ndarray, mask=None, best_fit=None, normalize=False):
        """
        Instantiate a MagData object

        Parameters
        ----------
        fid_file_path : Path
            Path to the original *.sdf file containing the experimental data
        algorithm :
            Algorithm used to extract the magnetization values
        mag_data : numpy.ndarray
            Magnetization data array
        B_relax : numpy.ndarray
            Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID
            data and to a 1D array of tau.
        tau : numpy.ndarray
            Array if explored incrementation times. Each element of this array corresponds to a 1D array of FID data.
        mask : numpy.ndarray of bool, optional
            Mask to apply on data array. Must be the same dimension as data array. Default is an array of False.
        best_fit : dict, optional
            Dictionary containing fitted model of the data. For each element of best_fit, the key is the name of the
            algorithm, while the value is a FitData or a FitDataArray object. An entry is added to the dictionary every
            time an algorithm is run on the data. Default is an empty dictionary.
        normalize : bool, optional
            Signal if the mag_data must be normalized or not. Default is False.
        """

        super().__init__(fid_file_path, "MAG", mag_data, mask, best_fit)

        self.algorithm = algorithm
        self.B_relax = B_relax
        self.tau = tau

        if normalize:
            self._normalize()

        self.update_mask(self.data <= 0)

    def _normalize(self):
        """
        Normalize the magnetization data so that it is in [0., 1] and decreasing.
        """
        output = []

        for mag_matrix_i, tau_i, mask_i in zip(self.data, self.tau, self.mask):
            unmask_data = mag_matrix_i[~mask_i]
            # cond = unmask_data[0] < unmask_data[-1]
            cond = tau_i[0] < tau_i[-1]
            m0 = unmask_data.min() if cond else unmask_data.max()
            m1 = unmask_data.min() if not cond else unmask_data.max()
            # m1 = unmask_data.max()
            output.append((m0 - mag_matrix_i) / (m0 - m1))

        output = np.array(output)
        output[output == 0.] = 1e-12  # Avoid errors with max_likelihood approaches

        self.data = output

    # TODO Implement __repr__

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
        mag_matrix = self.data[sl]
        B_relax = self.B_relax[sl[0]]
        tau = self.tau[sl]
        mask = self.mask[sl]
        best_fit = {key: value[sl] for key, value in self.best_fit.items()}

        return MagData(data_file_path, algorithm, mag_matrix, B_relax, tau, mask, best_fit)

    def __str__(self):
        output = ("-" * 16 + f'REPORT: MAG data matrix' + "-" * 16 + "\n" +
                  f"Data file path:             \t{self.data_file_path}\n" +
                  f"Experience name:            \t{self.experience_name}\n" +
                  f"Output save path:           \t{self.saving_folder}\n" +
                  f"Total mag matrix size:       \t({self.data.shape[0]} x {self.data.shape[1]}) " +
                  f"ou {np.prod(self.data.shape):,} points\n" +
                  f"Evolution fields (B_relax): \t{len(self.B_relax)} fields studied between "
                  f"{np.min(self.B_relax):.2e} and {np.max(self.B_relax):.2e} MHz\n" +
                  f"Incrementation time (tau):      \t{self.tau.shape[1]} pts per evolution field\n"
                  )
        for i, (B_i, tau_i) in enumerate(zip(self.B_relax, self.tau)):
            output += f"\t\t\t{i + 1}) B_relax={B_i:2.2e} MHz\t\ttau: [{tau_i.min():.2e} to {tau_i.max():.2e}] ms\n"
        output += f"Mask size:                  \t{np.sum(self.mask):,}/{np.prod(self.data.shape):,} pts"
        return output

    def apply_mask(self, sigma=2., dims="xy", display_report=False) -> None:
        """ Mask aberrant values in data

        Find and mask aberrant values in the fid data matrix. Aberrant values are defined as either negative values
        or discontinuities in the signal. Discontinuities in the data are found by evaluating the normalized gradient
        of the signal toward all chosen dimensions, and rejecting the data points whose normalized gradient magnitude is
        greater than a chosen sigma.

        The normalized gradient towards the i th  dimension is defined as (grad_i(FID) - mean(grad_i(FID))/
        std(grad_i(FID).

        If display_masked_data is True, then the data result of displayed to the user.

        Parameters
        ----------
        sigma : float
            Rejection criterion for the data points. Default is 2.
        dims : str, optional
            Specify which dimension(s) to use to find the aberrant data. Default is "xy".
        display_report : bool, optional
            Display the results of masking procedure. Default is False.

        Returns
        -------
        None
        """
        # Apply a mask aberrant data
        ind = self.data <= 0.
        self.mask[ind] = True

        grad = np.array(np.gradient(self.data))
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
            mag_matrix = ma.masked_array(self.data, mask=self.mask)
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

    def batch_plot(self, title="") -> None:
        """
        Plot all magnetization data in a single figure

        This methods plots every magnetization data array in a single figure with n axis, where n is the number of
        B_relax fields in the experiment.

        Parameters
        ----------
        title : str, optional
            Figure sup title. Default is "".

        Returns
        -------
        None
        """
        mag_data = ma.masked_array(self.data, self.mask)
        n_fig = len(self.B_relax)
        n_rows, n_columns = ceil(n_fig / 5), 5
        fig, axes_2D = plt.subplots(n_rows, n_columns, sharey="all",
                                    figsize=(20 / 2.54, n_rows * 5 / 2.54),
                                    squeeze=False)
        axes_1D = np.array(axes_2D).flatten()

        colormap = plt.cm.get_cmap("plasma")
        custom_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.tau.T))])

        for i, ax in enumerate(axes_1D):
            ax.set_prop_cycle(cycler('color', custom_colors))
            if i < len(mag_data):
                cst = np.max(np.absolute(mag_data[i]))
                ax.plot(self.tau[i], mag_data[i] / cst, "--.")
                ax.set_xscale("log")
                ax.tick_params(axis='both', which='major', pad=0)

                x_txt = self.tau[i].min()
                y_txt = mag_data[i].max()*0.9
                s = r"$B_{relax}$=" f"{self.B_relax[i]:.1e} MHz"
                ax.text(x_txt, y_txt, s)
            else:
                ax.axis('off')
        for axes_i in axes_2D:
            axes_i[0].set_ylabel("Mag. [arb. units]")

        for ax in axes_2D[-1]:
            ax.set_xlabel(r"Time $\tau$ [us]")
        fig.suptitle(title)
        plt.subplots_adjust(wspace=0)
        plt.show()

    def save_to_pdf(self, fit_to_plot=None, display=False):
        """
        Produce a pdf with the magnetization data

        This methods plots all magnetization curves into a single pdf file. Two figures are produced for each
        magnetization curve. The first figure display the magnetization and all previously computed fit (mono-exp and
        bi-exp models). If no fit method was called, then no fitted data is shown in the pdf. The second figure shows
        the error (residue) per point between the data and the adjusted models, if any. Using the fit_to_plot
        parameter, it is possible to specify one/multiple fit to plot in the figure, given that the relevant
        algorithm(s) was(were) applied to the data.

        Parameters
        ----------
        fit_to_plot : str or list of str, optional
            Key of the fit, or list of keys of the fit to plot to the FID data (see the appropriate algorithm to see
            the list of accessible keys). Default is "all".
        display: bool, optional
            Signal that specifies whether the figures are shown or not to the user. Default it False
        """

        file_name = f"{self.experience_name}_Magnetization.pdf"
        file_path = self.saving_folder / file_name
        title = f"{self.experience_name} - Magnetization"

        # Process the fit_to_plot keywords list
        if fit_to_plot is None:
            best_fit_keys = self.best_fit.keys()
        elif isinstance(fit_to_plot, str):
            best_fit_keys = [fit_to_plot] if fit_to_plot in self.best_fit else []
        else:
            best_fit_keys = []
            for key in fit_to_plot:
                best_fit_keys.append(key) if key in self.best_fit else None

        mag_matrix = ma.masked_array(self.data, self.mask)

        pdf = PDFSaver(file_path, 2, 4, title, True)
        for i, (tau_i, mag_i, B_relax_i) in enumerate(zip(self.tau, mag_matrix, self.B_relax)):
            ax = pdf.get_ax()
            ax.set_xlabel('$tau$ u.a.', fontsize=8)
            ax.plot(tau_i, mag_i, '*', markersize=5,
                    label=r"$B_{relax}$" + f" = {B_relax_i:.2e} MHz")

            for algo in best_fit_keys:
                fit = self.best_fit[algo][i]
                ax.plot(tau_i, fit.data, '--', markersize=1, lw=3, **fit.plot_keywords)

            ax.legend(loc="lower left", fontsize='xx-small', handlelength=1, handletextpad=0.2)
            ax.set_xscale('log')
            ax.grid()

            ax = pdf.get_ax()
            ax.set_xlabel('$tau$ u.a.', fontsize=8)
            ax.set_ylabel('Residue', fontsize=8)
            for algo in best_fit_keys:
                fit = self.best_fit[algo][i]
                ax.plot(tau_i, mag_i - fit.data, '*', lw=3, markersize=4, **fit.plot_keywords)
            ax.set_xscale('log')
            ax.grid()
        pdf.close_pdf()

    @staticmethod
    def _model_mono_exp(tau, amp, C, R1):
        """
        Mono-exponential model

        Parameters
        ----------
        tau : float or numpy.ndarray
            Incrementation time
        amp : float
            Scaling parameter for the exponential. Assuming that the data were properly normalized, this parameter
            should be close to 1.
        C : float
            Line base. Assuming that the data were properly normalized, this parameter should be close to 0.
        R1 : float
            Relaxation time

        Returns
        -------
        model_output : float or numpy.ndarray
            Computed output of the model
        """
        return amp * np.exp(-R1 * tau) + C

    @staticmethod
    def _model_bi_exp(tau, amp, alpha, C, R11, R12):
        """
        Bi-exponential model

        Parameters
        ----------
        tau : float or numpy.ndarray
            Incrementation time
        amp : float
            Scaling parameter for the exponential. Assuming that the data were properly normalized, this parameter
            should be close to 1.
        alpha : float
            Population of the first compartment. This parameter should be between 0. and 1.
        C : float
            Line base. Assuming that the data were properly normalized, this parameter should be close to 0.
        R11 : float
            Relaxation time of the first population
        R12 : float
            Relaxation time of the second population

        Returns
        -------
        model_output : float or numpy.ndarray
            Computed output of the model
        """
        return amp * alpha * np.exp(-R11 * tau) + amp * (1 - alpha) * np.exp(-R12 * tau) + C

    @staticmethod
    def _fit_mono_exp(tau, mag, mask):
        """
        Encapsulate the mono-exponential model

        Parameters
        ----------
        tau : numpy.ndarray
            Incrementation times
        mag : numpy.ndarray
            Magnetization data array
        mask : numpy.ndarray of bool, optional
            Mask to apply on data array

        Returns
        -------
        result_mono
            raw output of the lmfit model after fitting
        """
        mod = Model(MagData._model_mono_exp, independent_vars=['tau'])
        params = Parameters()
        R1 = 1 / tau[np.argmin(np.absolute(mag - 0.63))]
        params.add('amp', value=1., min=0.)
        params.add('R1', value=R1, min=0)
        params.add('C', value=0)
        ind = ~ mask
        result_mono = mod.fit(mag[ind], params, tau=tau[ind])
        result_mono.best_fit = MagData._model_mono_exp(tau, result_mono.params['amp'].value,
                                                       result_mono.params['C'].value,
                                                       result_mono.params['R1'].value, )
        return result_mono

    @staticmethod
    def _adjust_bi_exp(tau, mag, mask):
        """
        Encapsulate the bi-exponential model

        Parameters
        ----------
        tau : numpy.ndarray
            Incrementation times
        mag : numpy.ndarray
            Magnetization data array
        mask : numpy.ndarray of bool, optional
            Mask to apply on data array

        Returns
        -------
        result_bi
            raw output of the lmfit model after fitting

        """
        mod = Model(MagData._model_bi_exp, independent_vars=['tau'])
        params = Parameters()
        R1 = 1 / tau[np.argmin(np.absolute(mag - 0.63))]
        params.add('amp', value=1., min=0.)
        params.add('alpha', value=0.5, min=0., max=1.)
        params.add('R11', value=R1, min=0)
        params.add('R12', value=R1 * 2, min=0)
        params.add('C', value=0)
        ind = ~mask
        result_bi = mod.fit(mag[ind], params, tau=tau[ind])
        result_bi.best_fit = MagData._model_bi_exp(tau, result_bi.params['amp'].value, result_bi.params['alpha'].value,
                                                   result_bi.params['C'].value,
                                                   result_bi.params['R11'].value, result_bi.params['R12'].value)
        return result_bi

    def to_rel(self):
        """
        Extract the relaxation time and populations from the magnetization data

        This method extract the relaxation time and populations from the magnetization data using a mono and
        bi-exponential models.

        Returns
        -------
        rel_data : RelData
            Relaxometry data R1, R11, R12, and alpha, extracted from the magnetization object.

        """
        result_mono = []
        result_bi = []
        for tau_i, mag_i, mask_i in zip(self.tau, self.data, self.mask):
            result_mono.append(self._fit_mono_exp(tau_i, mag_i, mask_i))
            result_bi.append(self._adjust_bi_exp(tau_i, mag_i, mask_i))

        # Create FitDataArray objects from the two models
        fit = []
        for res_i in result_mono:
            fit.append(FitData(res_i.best_fit,
                               c='tab:pink',
                               label=r"$R_{1}$" + f"= {res_i.params['R1'].value:.2f}"
                               ))
        mono_best_fit = FitDataArray(fit)

        fit = []
        for res_i in result_bi:
            fit.append(FitData(res_i.best_fit,
                               c='tab:olive',
                               label=(r"$R_1^{(1)}$" + f"={res_i.params['R11'].value:.2f}\n"
                                                       r"$R_1^{(2)}$" + f"={res_i.params['R12'].value:.2f}")
                               ))
        bi_best_fit = FitDataArray(fit)

        self.best_fit["mono_exp"] = mono_best_fit
        self.best_fit["bi_exp"] = bi_best_fit

        # Format the model results to construct the relaxometry Data object
        R1 = np.array([res_i.params['R1'] for res_i in result_mono])
        R11 = np.array([res_i.params['R11'] for res_i in result_bi])
        R12 = np.array([res_i.params['R12'] for res_i in result_bi])
        alpha = np.array([res_i.params['alpha'] for res_i in result_bi])

        rel_matrix = np.array((R1, R11, R12, alpha))

        # mono_mask = np.array([[~res_i['success'] for res_i in result_mono]], dtype=bool)
        # bi_mask = np.array([[~res_i['success'] for res_i in result_bi]], dtype=bool)
        # rel_mask = np.append(mono_mask, bi_mask, bi_mask, bi_mask)

        return RelData(self.data_file_path, rel_matrix, self.B_relax)  # rel_mask)

    def to_ilt(self, R1: np.ndarray, lmd: float, reg_order=2, penalty="up"):
        ALPHA = []
        FIT = []
        R1_array = []
        LMD = []
        for tau_i, mag_i in zip(self.tau, self.data):
            alpha, fit, loss = utils.ilt_uniform_penalty(tau_i, mag_i+1., R1, lmd, reg_order, penalty)
            ALPHA.append(alpha)
            FIT.append(fit)
            R1_array.append(R1)
            LMD.append(lmd)

        LMD = np.array(LMD)
        R1_array = np.array(R1_array)

        fit = np.array(FIT)
        alpha = np.array(ALPHA)

        best_fit = FitData(fit-1,
                           c='tab:gray',
                           label=r"Inv. Laplace Transform")
        self.best_fit["ilt"] = best_fit

        return ILTData(self.data_file_path, "ILT", alpha, self.B_relax, R1_array, LMD)



