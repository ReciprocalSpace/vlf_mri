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


class FidData(VlfData):
    def __init__(self, fid_file_path: Path, fid_matrix, B_relax, tau, t_fid, mask=None, best_fit=None) -> None:
        if best_fit is None:
            best_fit = {}

        super().__init__(fid_file_path, "FID", best_fit)

        self.fid_matrix = fid_matrix
        if mask is None:
            self.mask = np.zeros_like(fid_matrix, dtype=bool)
        else:
            self.mask = mask

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

        sdf_file_path = self.sdf_file_path
        fid_matrix = self.fid_matrix[sl]
        B_relax = self.B_relax[sl[0]]
        tau = self.tau[sl[0:2]]
        t_fid = self.t_fid[sl[-1]]
        mask = self.mask[sl]
        best_fit = {key: value(item)[sl] for key, value in self.best_fit}

        return FidData(sdf_file_path, fid_matrix, B_relax, tau, t_fid, mask, best_fit)

    def __str__(self):
        output = ("-" * 16 + f'REPORT: FID data matrix' + "-" * 16 + "\n" +
                  f"SDF file path:              \t{self.sdf_file_path}\n" +
                  f"Experience name:            \t{self.experience_name}\n" +
                  f"Output save path:           \t{self.saving_folder}\n" +
                  "Total fid matrix size:       \t" +
                  f"({self.fid_matrix.shape[0]} x {self.fid_matrix.shape[1]} x {self.fid_matrix.shape[2]}) " +
                  f"ou {np.prod(self.fid_matrix.shape):,} points\n" +
                  f"Champs evolution (B_relax): \t{len(self.B_relax)} champs étudiés entre {np.min(self.B_relax):.2e} " +
                  f"et {np.max(self.B_relax):.2e} MHz\n" +
                  f"Temps evolution (tau):      \t{self.tau.shape[1]} pts par champ d evolution\n"
                  )

        for i, (B_i, tau_i) in enumerate(zip(self.B_relax, self.tau)):
            output += f"\t\t\t{i + 1}) B_relax={B_i:2.2e} MHz\t\ttau: [{tau_i.min():.2e} à {tau_i.max():.2e}] ms\n"
        output += (f"Signal ind libre (FID):     \t{self.fid_matrix.shape[-1]} pts espacés de " +
                   f"{self.t_fid[1] - self.t_fid[0]:.2e} us par FID\n")
        output += f"Mask size:                  \t{np.sum(self.mask):,}/{np.prod(self.fid_matrix.shape):,} pts"
        return output

    def __repr__(self):
        return f"vlf_mri.FidData(Path('{self.sdf_file_path}'), fid_matrix, B_relax, tau, t_fid, mask, best_fit)"

    def apply_mask(self, sigma=2., dims="xyz", display_report=False) -> None:
        """ Mask aberrant values in fid_matrix

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
        ind = self.fid_matrix <= 0.
        self.mask[ind] = True

        grad = np.array(np.gradient(self.fid_matrix))
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
            fid_matrix = ma.masked_array(self.fid_matrix, mask=self.mask)
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

    def batch_plot(self, suptitle="") -> None:
        fid_matrix = ma.masked_array(self.fid_matrix, self.mask)
        n_fig = len(self.B_relax)
        n_rows, n_columns = ceil(n_fig / 3), 3
        fig, axes_2D = plt.subplots(n_rows, n_columns, sharex='all', sharey="all",
                                    figsize=(16 / 2.54, n_rows * 5 / 2.54),
                                    squeeze=False)
        axes_1D = np.array(axes_2D).flatten()

        colormap = plt.cm.get_cmap("plasma")
        costum_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.tau.T))])

        # costum_cycler = cycler(color=[colormap(x) for x in np.linspace(0, 0.8, len(tau))])
        for i, ax in enumerate(axes_1D):
            ax.set_prop_cycle(cycler('color', costum_colors))
            if i < len(fid_matrix):
                cst = np.max(np.absolute(fid_matrix[i]))
                ax.plot(self.t_fid, fid_matrix[i].T / cst)
                ax.tick_params(axis='both', which='major', pad=0)
                ax.text(0, 0.9, r"$B_{relax}$=" f"{self.B_relax[i]:.1e} MHz")
            else:
                ax.axis('off')
        for axes_i in axes_2D:
            axes_i[0].set_ylabel("Signal [arb. units]")

        for ax in axes_2D[-1]:
            ax.set_xlabel("Time [us]")
        fig.suptitle(suptitle)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def save_to_pdf(self, fit_to_plot=None, display=False) -> None:
        print("-" * 16 + "saving FID matrix to PDF" + "-" * 16)

        if fit_to_plot is None:
            best_fit_keys = self.best_fit.keys()
        elif isinstance(fit_to_plot, str):
            best_fit_keys = [fit_to_plot] if fit_to_plot in self.best_fit else []
        else:
            best_fit_keys = []
            for key in fit_to_plot:
                best_fit_keys.append(key) if key in self.best_fit else None

        fid_matrix = ma.masked_array(self.fid_matrix, self.mask)
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
                    ax.plot(self.t_fid, fit[i, j], '.r', markersize=1, label=algo)
                ax.legend(loc="lower left", fontsize=8, handlelength=0.5, handletextpad=0.2)
                ax.grid()
            pdf_saver.close_pdf()

    def to_mag_mean(self, t_0=1., t_1=25.) -> MagData:
        index_min = np.argmin(np.absolute((self.t_fid - t_0)))
        index_max = np.argmin(np.absolute((self.t_fid - t_1)))

        mean_mag = np.mean(self.fid_matrix[:, :, index_min:index_max], axis=2)

        best_fit = np.expand_dims(mean_mag, axis=2)
        best_fit = np.tile(best_fit, (1, 1, len(self.t_fid)))

        mask = np.ones_like(self.fid_matrix, dtype=bool)
        mask[:, :, index_min:index_max] = False

        best_fit = ma.array(best_fit, mask=mask)

        self.best_fit["mean"] = best_fit

        return MagData(self.sdf_file_path, "mean", mean_mag, self.B_relax, self.tau)

    def to_mag_intercept(self, t_0=1., t_1=25.) -> MagData:
        fid_matrix = self.fid_matrix
        mask = self.mask

        index_min = np.argmin(np.absolute((self.t_fid - t_0)))
        index_max = np.argmin(np.absolute((self.t_fid - t_1)))

        shape = self.fid_matrix.shape
        nx, ny, nz = shape

        len_fid = index_max - index_min

        Y = fid_matrix[:, :, index_min:index_max].reshape((-1, len_fid)).T
        Y_mask = mask[:, :, index_min:index_max].reshape((-1, len_fid)).T

        Y = ma.masked_array(Y, Y_mask)
        X = np.tile(np.arange(index_min, index_max), (Y.shape[1], 1)).T
        X = ma.masked_array(X, Y_mask)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        Y_mean = Y.mean(axis=0)

        a = (np.mean(X * Y, axis=0) - X_mean * Y_mean) / X_std ** 2
        b = Y_mean - a * X_mean

        # # Construction du best_fit
        X = np.arange(fid_matrix.shape[-1])
        best_fit = []
        for a_i, b_i in zip(a, b):
            best_fit.append(X * a_i + b_i)

        mask = np.ones(fid_matrix.shape[-1], dtype=bool)
        mask[index_min:index_max] = False
        mask = np.tile(mask, (*fid_matrix.shape[0:2], 1))
        best_fit = ma.masked_array(best_fit, mask=mask)
        best_fit = best_fit.reshape(fid_matrix.shape)

        self.best_fit["intercept"] = best_fit

        intercept = b.reshape((nx, ny))

        return MagData(self.sdf_file_path, "intercept", intercept, self.B_relax, self.tau)

    def max_likelihood(self) -> MagData:
        def likelihood(theta: list, *args):
            M_0, T2, sigma = theta
            t, noisy_signal = args
            signal = M_0 * np.exp(-t / T2)
            log_likelihood = rice.logpdf(noisy_signal, signal / sigma, loc=0, scale=sigma)
            return -np.sum(log_likelihood)

        def model(theta, t_fid):
            M_0, T2, sigma = tuple(theta)
            S = M_0 * np.exp(-t_fid / T2)
            SNR = S / sigma
            index = (SNR < 37)
            not_index = np.logical_not(index)

            out = np.zeros_like(t_fid)

            out[index] = rice.stats(SNR[index], scale=sigma, moments='m')
            out[not_index] = S[not_index]
            return out

        tau = self.tau
        magnetization = []
        best_fit = []
        mask = []
        for fid_i in self.fid_matrix:
            magnetization_i = []
            best_fit_i = []
            mask_i = []
            for fid_ij in fid_i:
                ind = ~self.mask
                unmask_fid = fid_ij[ind]
                # bounds = ((1, np.max(fid_ij)), (1e-6, None), (0, None)) # Pas appli
                M_0 = np.mean(unmask_fid[:50])
                sigma = unmask_fid[-30:].std() ** 2
                T_2 = M_0 * (
                        fid_ij.max() - fid_ij.min()) ** -1 / 2  # T2 = M0 / slope en zero (/2 juste pour le réduire un peu)

                x0 = np.array([M_0, T_2, sigma])
                result = minimize(likelihood, x0=x0, args=(tau[ind], unmask_fid + 1e-3), method='Nelder-Mead')
                magnetization_i.append(result['x'][0])
                best_fit_i.append(model(result['x'], tau))
                data_point_to_be_mask = not result['success'] or not np.alltrue(result['x'] > 0.)
                mask_i.append(data_point_to_be_mask)

                if data_point_to_be_mask:
                    print("DATA MASKED")
                    print("x0", x0)
                    print("x1", result["x"])
            magnetization.append(magnetization_i)
            best_fit.append(best_fit_i)
            mask.append(mask_i)

        return ma.array(magnetization, mask=np.array(mask, dtype=bool)), np.array(best_fit)