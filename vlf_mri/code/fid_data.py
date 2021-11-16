import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from cycler import cycler
from math import ceil
from numpy import ma
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rice
from tqdm import tqdm

from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData
from vlf_mri.code.mag_data import MagData
from vlf_mri.code.fit_data import FitData


class FidData(VlfData):
    """
    Contain and manage Free Induction Decay data

    This class stores and manages Free Induction Decay data.

    Attributes
    ----------
    B_relax : numpy.ndarray
        Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID data
        and to a 1D array of tau.
    tau : numpy.ndarray
        Array if explored incrementation times. Each element of this array corresponds to a 1D array of FID data.
    t_fid : numpy.ndarray
        Array of experimental time. Each element of this array corresponds to a data point in the FID data. This array
        is 1D, as it is assumed all FID arrays were acquired using the same number of points measured on the same
        duration.

    Methods
    -------
    apply_mask(sigma, dims, display_report):
        Mask aberrant values in data
    batch_plot(suptitle):
    save_to_pdf(fit_to_plot, display):
    to_mag_mean(self, t_0=1., t_1=25.)

    """
    def __init__(self, fid_file_path: Path, fid_data: np.ndarray, B_relax: np.ndarray, tau: np.ndarray,
                 t_fid: np.ndarray, mask=None, best_fit=None) -> None:

        if mask is None:
            mask = np.zeros_like(fid_data, dtype=bool)

        if best_fit is None:
            best_fit = {}

        super().__init__(fid_file_path, "FID", fid_data, mask, best_fit)

        self.B_relax = B_relax
        self.tau = tau
        self.t_fid = t_fid

    def __getitem__(self, item):
        # Defines the slices objects used to slice through the objects
        sl = []
        if not isinstance(item, Iterable):  # 2D or 3D slices
            item = (item,)
        for it in item:
            sl.append(it if isinstance(it, slice) else slice(it, it + 1))
        while len(sl) < 3:
            sl.append(slice(0, None))

        sl = tuple(sl)

        data_file_path = self.data_file_path
        fid_matrix = self.data[sl]
        B_relax = self.B_relax[sl[0]]
        tau = self.tau[sl[0:2]]
        t_fid = self.t_fid[sl[-1]]
        mask = self.mask[sl]
        best_fit = {key: value[sl] for key, value in self.best_fit.items()}

        return FidData(data_file_path, fid_matrix, B_relax, tau, t_fid, mask, best_fit)

    def __str__(self):
        output = ("-" * 16 + f'REPORT: FID data matrix' + "-" * 16 + "\n" +
                  f"Data file path:             \t{self.data_file_path}\n" +
                  f"Experience name:            \t{self.experience_name}\n" +
                  f"Output save path:           \t{self.saving_folder}\n" +
                  "Total fid matrix size:       \t" +
                  f"({self.data.shape[0]} x {self.data.shape[1]} x {self.data.shape[2]}) " +
                  f"ou {np.prod(self.data.shape):,} points\n" +
                  f"Champs evolution (B_relax): \t{len(self.B_relax)} champs étudiés entre {np.min(self.B_relax):.2e} " +
                  f"et {np.max(self.B_relax):.2e} MHz\n" +
                  f"Temps evolution (tau):      \t{self.tau.shape[1]} pts par champ d evolution\n"
                  )

        for i, (B_i, tau_i) in enumerate(zip(self.B_relax, self.tau)):
            output += f"\t\t\t{i + 1}) B_relax={B_i:2.2e} MHz\t\ttau: [{tau_i.min():.2e} à {tau_i.max():.2e}] ms\n"
        output += (f"Signal ind libre (FID):     \t{self.data.shape[-1]} pts espacés de " +
                   f"{self.t_fid[1] - self.t_fid[0]:.2e} us par FID\n")
        output += f"Mask size:                  \t{np.sum(self.mask):,}/{np.prod(self.data.shape):,} pts"
        return output

    def __repr__(self):
        return f"vlf_mri.FidData(Path('{self.data_file_path}'), data, B_relax, tau, t_fid, mask, best_fit)"

    def apply_mask(self, sigma=2., dims="xyz", display_report=False) -> None:
        """ Mask aberrant values in data

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
        """
        # Apply a mask aberrant data
        ind = self.data <= 0.
        self.mask[ind] = True

        grad = np.array(np.gradient(self.data))
        # Normalize gradient array
        axis = (1, 2, 3)
        grad_mean = np.expand_dims(grad.mean(axis=axis), axis=axis)
        grad_std = np.expand_dims(grad.std(axis=axis), axis=axis)
        grad = (grad - grad_mean) / grad_std

        # grad_n = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        grad_B = grad[0] if 'x' in dims else 0
        grad_tau = grad[1] if 'y' in dims else 0
        grad_t = grad[2] if 'z' in dims else 0
        grad_n = np.sqrt(grad_B ** 2 + grad_tau ** 2 + grad_t ** 2)

        self.mask = np.logical_or(self.mask, grad_n > sigma)

        if display_report:
            fid_matrix = ma.masked_array(self.data, mask=self.mask)
            fig, (axes_bef, axes_after) = plt.subplots(2, 4, tight_layout=True, figsize=(15 / 2.54, 10 / 2.54))

            fig.suptitle("Report: Apply mask")
            grad_B, grad_tau, grad_t = tuple(grad.reshape((3, -1)))
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
            # Second ax
            axes_bef[1].plot(grad_B[~ind], grad_t[~ind], ".", c="b", label="_")
            axes_bef[1].plot(grad_B[ind], grad_t[ind], ".", c="r", label="_")
            axes_bef[1].set_xlabel(r'Gradient $\nabla_B$(FID)', labelpad=0)
            axes_bef[1].set_ylabel(r'Gradient $\nabla_t$(FID)', labelpad=0)
            # Third ax
            axes_bef[2].plot(grad_tau[~ind], grad_t[~ind], ".", c="b", label="_")
            axes_bef[2].plot(grad_tau[ind], grad_t[ind], ".", c="r", label="_")
            axes_bef[2].set_xlabel(r'Gradient $\nabla_\tau$(FID)', labelpad=0)
            axes_bef[2].set_ylabel(r'Gradient $\nabla_t$(FID)', labelpad=0)
            # Fourth ax
            axes_bef[3].hist(np.log(grad_n + 1), 30, density=True)  # +1 to avoid log(0)
            axes_bef[3].set_xlabel(r"Norm $\log(\|\nabla\|)$")
            axes_bef[3].set_ylabel(r"Point density")
            # Second row
            # First ax
            axes_after[0].plot(grad_B[~ind], grad_tau[~ind], ".", c="b", label="_")
            axes_after[0].set_xlabel(r'Gradient $\nabla_B$(FID)', labelpad=0)
            axes_after[0].set_ylabel("AFTER\nGradient " r"$\nabla_\tau$(FID)", labelpad=0)
            # Second ax
            axes_after[1].plot(grad_B[~ind], grad_t[~ind], ".", c="b", label="_")
            axes_after[1].set_xlabel(r'Gradient $\nabla_B$(FID)', labelpad=0)
            axes_after[1].set_ylabel(r'Gradient $\nabla_t$(FID)', labelpad=0)
            # Third ax
            axes_after[2].plot(grad_tau[~ind], grad_t[~ind], ".", c="b", label="_")
            axes_after[2].set_xlabel(r'Gradient $\nabla_\tau$(FID)', labelpad=0)
            axes_after[2].set_ylabel(r'Gradient $\nabla_t$(FID)', labelpad=0)
            # Fourth ax
            axes_after[3].hist(np.log(grad_n[~ind] + 1), 30, density=True)  # +1 to avoid log(0)
            axes_after[3].set_xlabel(r"Norm $\log(\|\nabla\|)$", labelpad=0)
            axes_after[3].set_ylabel("Point density", labelpad=0)
            # Finishing legend and labels...
            lines_labels = [ax.get_legend_handles_labels() for ax in np.array(fig.axes).flatten()]
            for ax in np.array(fig.axes).flatten():
                ax.tick_params(axis='both', which='major', pad=0)
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title=f"{nb_pts} data points")

            plt.show()

            self.batch_plot("Report: FID after mask")

    def batch_plot(self, title="") -> None:
        """
        Plot all FID data in a single figure

        This methods plots every FID array in a single figure with n axis, where n is the number of B_relax fields in
        the experiment.

        Parameters
        ----------
        title : str, optional
            Figure sup title. Default is "".

        Returns
        -------
        None
        """
        fid_matrix = ma.masked_array(self.data, self.mask)

        n_fig = len(self.B_relax)
        n_rows, n_columns = ceil(n_fig / 3), 3

        fig, axes_2D = plt.subplots(n_rows, n_columns, sharex='all', sharey="all",
                                    figsize=(16 / 2.54, n_rows * 5 / 2.54),
                                    squeeze=False)
        axes_1D = np.array(axes_2D).flatten()

        # Produce a smooth color cycler for successive curves in an axis. This helps the user assess whether two
        # successive FID are close on the image. This helps quickly see if there is a problem in the data.
        colormap = plt.cm.get_cmap("plasma")
        custom_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.tau.T))])
        for i, ax in enumerate(axes_1D):
            ax.set_prop_cycle(cycler('color', custom_colors))
            if i < len(fid_matrix):
                cst = np.max(np.absolute(fid_matrix[i]))  # Curve normalization
                ax.plot(self.t_fid, fid_matrix[i].T / cst)
                ax.tick_params(axis='both', which='major', pad=0)
                ax.text(0, 0.9, r"$B_{relax}$=" f"{self.B_relax[i]:.1e} MHz")
            else:
                ax.axis('off')
        for axes_i in axes_2D:
            axes_i[0].set_ylabel("Signal [arb. units]")  # Display y label only for first axis of each row

        for ax in axes_2D[-1]:
            ax.set_xlabel("Time [us]")  # Display x label only for last axes row

        fig.suptitle(title)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def save_to_pdf(self, fit_to_plot="all", display=False) -> None:
        """
        Produce a pdf with the FID signal for every relaxation field

        This methods plots every FID signal for a given B_relax relax to a single axis and produces as many pdf file
        as there are B_relax fields. By default, every previously computed fit of the FID data are also plotted on the
        figures. If no fit method were called, then no fitted data is shown in the pdf. Using the fit_to_plot
        parameter, it is possible to specify one/multiple fit to plot in the figure, given that the relevant
        algorithm(s) was(were) applied to the data.

        Parameters
        ----------
        fit_to_plot : str or list of str, optional
            Key of the fit, or list of keys of the fit to plot to the FID data (see the appropriate algorithm to see
            the list of accessible keys). Default is "all".
        display: bool, optional
            Signal that specifies whether the figures are shown or not to the user. Default it False

        Returns
        -------
        None
        """
        print("-" * 16 + "saving FID matrix to PDF" + "-" * 16)

        # Generate/validate the list of fit to plot
        if fit_to_plot == "all":
            best_fit_keys = self.best_fit.keys()
        elif isinstance(fit_to_plot, str):
            best_fit_keys = [fit_to_plot] if fit_to_plot in self.best_fit else []
        else:
            best_fit_keys = []
            for key in fit_to_plot:
                best_fit_keys.append(key) if key in self.best_fit else None

        fid_matrix = ma.masked_array(self.data, self.mask)
        for i, (B_relax_i, tau_i, fid_i) in enumerate(zip(self.B_relax, self.tau, fid_matrix)):
            file_name = f"{self.experience_name}_FID_{int(B_relax_i)}-{int((B_relax_i % 1) * 1000):03}MHz.pdf"
            file_path = self.saving_folder / file_name

            print(f"\tSaving pdf: {file_path}")

            suptitle = (f"{self.experience_name} - FID, " + r"$B_{relax}$=" + f"{B_relax_i:.2f}MHz")

            pdf_saver = PDFSaver(file_path, 4, 5, title=suptitle, display_pages=display)
            for j, (tau_ij, fid_ij) in tqdm(enumerate(zip(tau_i, fid_i)), total=len(fid_i)):
                ax = pdf_saver.get_ax()
                ax.set_xlabel(r'Temps $t$ [$\mu$s]', fontsize=8)
                ax.plot(self.t_fid, fid_ij, '.', markersize=1, label=r"$\tau$" + f"= {tau_ij:.2e}")

                for algo in best_fit_keys:
                    fit = self.best_fit[algo]
                    fit_data = fit[i, j]
                    ax.plot(self.t_fid, ma.masked_array(fit_data.data, fit_data.mask), '.r', markersize=1,
                            **fit_data.plot_keywords)
                ax.legend(loc="lower left", fontsize=8, handlelength=0.5, handletextpad=0.2)
                ax.grid()
            pdf_saver.close_pdf()

    def to_mag_mean(self, t_0=1., t_1=25.) -> MagData:
        """
        Extract the magnetization for the FID signal using the "mean" algorithm

        This methods computes the mean value of the FID signal for all data points between t_0 and t_1 [us] and produces
        a MagData object.

        The result of the fit is also added to the best_fit dictionary attribute with "mean" key.

        Parameters
        ----------
        t_0 : float, optional
            Beginning of the domain used for averaging. Default is 1. us.
        t_1 : float, optional
            End of the domain used for averaging. Default is 25. us.

        Returns
        -------
        mag_data : MagData
            Object containing the magnetization data extracted from the FID signal.
        """
        index_min = np.argmin(np.absolute((self.t_fid - t_0)))
        index_max = np.argmin(np.absolute((self.t_fid - t_1)))

        mean_mag = np.mean(self.data[:, :, index_min:index_max], axis=2)

        # Best fit data must have the same dimension as FID data, but we only want to display the fit over the data
        # points used in the 'mean' algorithm -> we have to mask the other points
        best_fit = np.expand_dims(mean_mag, axis=2)
        best_fit = np.tile(best_fit, (1, 1, len(self.t_fid)))
        mask = np.ones_like(self.data, dtype=bool)
        mask[:, :, index_min:index_max] = False
        fit_data = FitData(best_fit, mask, label=f"Mean")
        self.best_fit["mean"] = fit_data

        return MagData(self.data_file_path, "mean", mean_mag, self.B_relax, self.tau, normalize=True)

    def to_mag_intercept(self, t_0=1., t_1=25.) -> MagData:
        """
        Extract the magnetization for the FID signal using the "intercept" algorithm

        This methods extract the magnetization value of the FID signal by computing the intercept of a the first
        order polynomial linear regression (y = a*x + b) over the data points between t_0 and t_1 [us]. It outputs a
        MagData object.

        The result of the fit is also added to the best_fit dictionary attribute with "intercept" key.

        Parameters
        ----------
        t_0 : float, optional
            Beginning of the domain used for averaging. Default is 1. us.
        t_1 : float, optional
            End of the domain used for averaging. Default is 25. us.

        Returns
        -------
        mag_data : MagData
            Object containing the magnetization data extracted from the FID signal.
        """
        fid_matrix = self.data
        mask = self.mask

        index_min = np.argmin(np.absolute((self.t_fid - t_0)))
        index_max = np.argmin(np.absolute((self.t_fid - t_1)))

        shape = self.data.shape
        nx, ny, nz = shape

        len_fid = index_max - index_min

        # Slice the FID signal and reshape it as a 2D matrix. This allows a vectorized computation of the linear
        # regression over every FIDé
        Y = fid_matrix[:, :, index_min:index_max].reshape((-1, len_fid)).T
        Y_mask = mask[:, :, index_min:index_max].reshape((-1, len_fid)).T

        Y = ma.masked_array(Y, Y_mask)
        # X are integers, and not t_fid, but this is unimportant, as we do not use the slope "a".
        X = np.tile(np.arange(index_min, index_max), (Y.shape[1], 1)).T
        X = ma.masked_array(X, Y_mask)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        Y_mean = Y.mean(axis=0)

        # Y = (a*X + b) where a and b are arrays with size equals to Y.shape[0]
        a = (np.mean(X * Y, axis=0) - X_mean * Y_mean) / X_std ** 2  # intercept
        b = Y_mean - a * X_mean  # slope

        # Best fit data must have the same dimension as FID data, but we only want to display the fit over the data
        # points used in the 'mean' algorithm -> we have to mask the other points
        X = np.arange(fid_matrix.shape[-1])
        best_fit = []
        for a_i, b_i in zip(a, b):
            best_fit.append(X * a_i + b_i)
        best_fit = np.array(best_fit)
        mask = np.ones(fid_matrix.shape[-1], dtype=bool)
        mask[index_min:index_max] = False
        mask = np.tile(mask, (*fid_matrix.shape[0:2], 1))
        best_fit = best_fit.reshape(fid_matrix.shape)
        fit_data = FitData(best_fit, mask, label="intercept")

        self.best_fit["intercept"] = fit_data

        intercept = b.reshape((nx, ny))

        return MagData(self.data_file_path, "intercept", intercept, self.B_relax, self.tau, normalize=True)

    def to_mag_max_likelihood(self) -> MagData:
        """
        Extract the magnetization for the FID signal using the "max_likelihood" algorithm

        This methods extract the magnetization value of the FID signal by maximizing the likelihood of observing a given
        FID experimental signal considering a specific data model. This current implementation assumes a Rician noise
        distribution with mean 0. and standard deviation sigma over a mono-exponential FID signal :
        M(t) = M0 * exp(-t/T2). In practice, this model is ofter too simple to accurately describe the measured FID
        signal, which can lead to important bias and errors in the evaluation of the magnetization.  It outputs a
        MagData object.

        The result of the fit is also added to the best_fit dictionary attribute with "max_likelihood" key.

        Returns
        -------
        mag_data : MagData
            Object containing the magnetization data extracted from the FID signal.
        """
        def likelihood(theta: list, *args):
            """
            Compute the likelihood of a signal given a model

            This function computes the likelihood of measured a given noisy signal considering a model.

            Parameters
            ----------
            theta : list
                List of parameters in the model. theta = [M_0, T2, sigma]
            args : tuple
                args = (t_fid, noisy_signal)

            Returns
            -------
            log_likelihood : numpy.ndarray
                Log likelihood (negative) of the noisy signal for the model. Note: the negative value is returned to
                make the output of this function compatible with the minimization method of the scipy library.
            """
            M_0, T2, sigma = theta
            t, noisy_signal = args
            signal = M_0 * np.exp(-t / T2)
            log_likelihood = rice.logpdf(noisy_signal, signal / sigma, loc=0, scale=sigma)
            return -np.sum(log_likelihood)

        def model(theta, t_fid):
            """
            Compute the expected signal considering a model

            Parameters
            ----------
            theta : list
                List of parameters in the model. theta = [M_0, T2, sigma]
            t_fid : numpy.ndarray
                Array of experimental time.

            Returns
            -------
            model : numpy.ndarray
                Computed expectancy signal
            """
            M_0, T2, sigma = tuple(theta)
            S = M_0 * np.exp(-t_fid / T2)
            SNR = S / sigma

            # Note : for SNR > 37, the rice.stats returns a div by zero error. This is a numerical issue in the
            # implementation of the rice package of the scipy library. For points with high SNR, the rician distribution
            # becomes close to the gaussian distribution with 0. mean, i.e. the noise does not influence the expected
            # value of the signal. This is not the case at low SNR.
            index = (SNR < 37)
            not_index = np.logical_not(index)

            out = np.zeros_like(t_fid)

            out[index] = rice.stats(SNR[index], scale=sigma, moments='m')
            out[not_index] = S[not_index]
            return out

        mag = []
        best_fit = []
        mag_mask = []
        # Iteration over the B_relax field values
        for fid_i, mask_i in zip(self.data, self.mask):
            mag_i = []
            best_fit_i = []
            mag_mask_i = []
            # Iteration over the tau values
            for fid_ij, mag_mask_ij in zip(fid_i, mask_i):
                ind = ~ mag_mask_ij
                fid = fid_ij[ind]
                t_fid = self.t_fid[ind]

                # Approximate initial parameter values for the minimization algorithm
                M_0 = np.mean(fid[:30])
                sigma = fid[-30:].std() ** 2
                T_2 = np.mean(t_fid[-30:]) / np.log(M_0/np.mean(t_fid[-30:]))
                x0 = np.array([M_0, T_2, sigma])

                result = minimize(likelihood, x0=x0, args=(self.t_fid[ind], fid + 1e-3), method='Nelder-Mead')
                mag_i.append(result['x'][0])

                best_fit_i.append(model(result['x'], self.t_fid))

                # Mask the point if there was an error in the minimization
                data_point_to_be_mask = not result['success'] or not np.alltrue(result['x'] > 0.)
                mag_mask_i.append(data_point_to_be_mask)

            mag.append(mag_i)
            best_fit.append(best_fit_i)
            mag_mask.append(mag_mask_i)

        mag = np.array(mag)
        best_fit = np.array(best_fit)
        mag_mask = np.array(mag_mask, dtype=bool)

        best_mask = np.expand_dims(mag_mask, 2)
        best_mask = np.tile(best_mask, (1, 1, len(self.t_fid)))
        fit_data = FitData(best_fit, best_mask, label="max likelihood")

        self.best_fit["max_likelihood"] = fit_data

        return MagData(self.data_file_path, "max_likelihood", mag, self.B_relax, self.tau, mag_mask, normalize=True)
