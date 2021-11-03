import numpy as np
from pathlib import Path

from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData


class RelData(VlfData):
    def __init__(self, data_file_path: Path, rel_data: np.ndarray,
                 B_relax: np.ndarray, mask=None, best_fit=None) -> None:

        super().__init__(data_file_path, "REL", rel_data, mask, best_fit)

        self.data = rel_data
        self.B_relax = B_relax

        self._reorder_R11_R12()

    def _reorder_R11_R12(self) -> None:
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
