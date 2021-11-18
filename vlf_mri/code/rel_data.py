import numpy as np
from pathlib import Path

from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData


class RelData(VlfData):
    """
    Contain and manage relaxometry data

    This class stores and manages relaxometry data. Please note that the data object in this class is set differently
    than for the other data VlfData classes.

    data[0] : R1 for a mono-exponential model
    data[1] : R11 for the bi-exponential model
    data[2] : R12 for the bi-exponential model
    data[3] : alpha, or population, associated with the first compartment  (R11)

    1-data[3] : population for the second compartment (R12)

    Attributes
    ----------
    B_relax : numpy.ndarray
        Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID
        data and to a 1D array of tau.

    Methods
    -------
    save_to_pdf(fit_to_plot, display):
        Produce a pdf with the relaxometry data

    """

    def __init__(self, fid_file_path: Path, rel_data: np.ndarray,
                 B_relax: np.ndarray, mask=None, best_fit=None) -> None:
        """
        Instantiate a RelData object

        Parameters
        ----------
        fid_file_path : Path
            Path to the original *.sdf file containing the experimental data
        rel_data : numpy.ndarray
            Relaxation data array. Please note that the data object in this class is set differently
            than for the other data VlfData classes.
            data[0] : R1 for a mono-exponential model
            data[1] : R11 for the bi-exponential model
            data[2] : R12 for the bi-exponential model
            data[3] : alpha, or population, associated with the first compartment  (R11)
        B_relax : numpy.ndarray
            Array of explored relaxation magnetic fields. Each element of this array corresponds to a 2D array of FID
            data and to a 1D array of tau.
        mask : numpy.ndarray of bool, optional
            Mask to apply on data array. Must be the same dimension as data array. Default is an array of False.
        best_fit : dict, optional
            Dictionary containing fitted model of the data. For each element of best_fit, the key is the name of the
            algorithm, while the value is a FitData or a FitDataArray object. An entry is added to the dictionary every
            time an algorithm is run on the data. Default is an empty dictionary.
        """

        super().__init__(fid_file_path, "REL", rel_data, mask, best_fit)

        # self.data = rel_data
        self.B_relax = B_relax

        self._reorder_R11_R12()

    def _reorder_R11_R12(self) -> None:
        """
        Classify R11 and R12 values based on their values

        Very simple implementation of a classifying algorithm for the triplet of R11, R12, alpha in data. For all
        B_relax values, this methods classify the (R11,R12) tuplets as R11'=min(R11,R12), R12'=max(R11,R12). This
        algorithm works only if there are no crossings of the R11 and R12 values in the data matrix.

        Returns
        -------

        """
        R11_R12 = self.data[1:3]
        alpha = self.data[3]

        ind = np.argsort(R11_R12, axis=0)
        R11_R12 = np.sort(R11_R12, axis=0)

        alpha = np.array([[a_i if ind_i == 0 else 1-a_i for a_i, ind_i in zip(alpha, ind[0])]])

        self.data[1:] = np.concatenate((R11_R12, alpha))

    # TODO implement __repr__

    def __str__(self):
        output = ("-" * 16 + f'REPORT: REL data matrix' + "-" * 16 + "\n" +
                  f"Data file path:             \t{self.data_file_path}\n" +
                  f"Experience name:            \t{self.experience_name}\n" +
                  f"Output save path:           \t{self.saving_folder}\n" +
                  f"Champs evolution (B_relax): \t{len(self.B_relax)} champs étudiés entre {np.min(self.B_relax):.2e}" +
                  f" et {np.max(self.B_relax):.2e} MHz\n"
                  )
        output += (f"Mask size:                  \t{np.sum(self.mask[0])+np.sum(self.mask[1]):,}/" +
                   f"{2*self.data.shape[1]:,} pts")
        return output

    def save_to_pdf(self, fit_to_plot=None, display=False) -> None:
        """
        Produce a pdf with the relaxation data

        This methods plots two figures into a pdf file. The first figure contains the relaxation rate R1 for the mono-
        exponential model, as well as the R11 and R12 values for the bi-exponential modes, as a function of the B_relax.
        The second figure displays the two populations alpha and 1-alpha, corresponding to the two compartments (R11
        and R12).

        Parameters
        ----------
        fit_to_plot : str or list of str, optional
            Key of the fit, or list of keys of the fit to plot to the FID data (see the appropriate algorithm to see
            the list of accessible keys). Default is "all". Note: this class actually contains no algorithm for data
            analysis. This parameter exist only for consistency with the other classes.
        display: bool, optional
            Signal that specifies whether the figures are shown or not to the user. Default it False
        """
        B_relax = self.B_relax
        R1 = self.data[0]
        R11 = self.data[1]
        R12 = self.data[2]
        alpha = self.data[3]

        # Create the pdfsaver object
        file_name = f"{self.experience_name}_Relaxation.pdf"
        file_path = self.saving_folder / file_name
        title = f"{self.experience_name} - Relaxation"

        pdf = PDFSaver(file_path, 1, 3, title, display)

        # First plot: R1, R11, R12 VS B_relax
        ax = pdf.get_ax()
        ax.plot(B_relax, R1, '--d', c="darkviolet", label=r'$R_1$')
        ax.plot(B_relax, R11, '--*', c='b', label=r'$R_1^{(1)}$')
        ax.plot(B_relax, R12, '--*', c='#0081FE', label=r'$R_1^{(2)}$')
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_title('Relaxation')

        # Second plot: alpha VS B_relax
        ax = pdf.get_ax()
        ax.plot(B_relax, alpha, '--*', c='b', label=r'$a_{bi}$')
        ax.plot(B_relax, 1-alpha, '--*', c='#0081FE', label=r'($1-a_{bi}$)')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.set_xlabel(r"$B_{relax}$  [MHz]")
        ax.set_title('population')

        pdf.close_pdf()
