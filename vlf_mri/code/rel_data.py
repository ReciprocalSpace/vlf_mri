import numpy as np
from vlf_mri.code.pdf_saver import PDFSaver
from vlf_mri.code.vlf_data import VlfData


class RelData(VlfData):
    def __init__(self, data_file_path, rel_data, B_relax, mask=None, best_fit=None):
        if best_fit is None:
            best_fit = {}

        super().__init__(data_file_path, "REL", best_fit)

        self.rel_matrix = rel_data
        self.B_relax = B_relax
        self.mask = np.zeros_like(rel_data, dtype=bool) if mask is None else mask


    def __str__(self):
        output = ("-" * 16 + f'REPORT: REL data matrix' + "-" * 16 + "\n" +
                  f"Data file path:             \t{self.data_file_path}\n" +
                  f"Experience name:            \t{self.experience_name}\n" +
                  f"Output save path:           \t{self.saving_folder}\n" +
                  f"Champs evolution (B_relax): \t{len(self.B_relax)} champs étudiés entre {np.min(self.B_relax):.2e}" +
                  f" et {np.max(self.B_relax):.2e} MHz\n"
                  )
        output += (f"Mask size:                  \t{np.sum(self.mask[0])+np.sum(self.mask[1]):,}/"+
                   f"{2*self.rel_matrix.shape[1]:,} pts")
        return output


    def plot_relaxation(self, result_ajust_mono, result_ajust_bi, B_relax, name_manip, folder):
        # Create the pdfsaver object
        file_name = f"{name_manip}_Relaxation.pdf"
        file_path = os.path.join(folder, file_name)
        title = f"{name_manip} - Magnetization"
        pdf = PDFSaver(file_path, 1, 3, title, True)

        # Prepare and sort the R1, R11, R12 and alpha data into arrays
        R1 = np.array([res.params['R1'].value for res in result_ajust_mono])
        R11_R12 = np.array([[res.params['R11'].value, res.params['R12'].value] for res in result_ajust_bi])
        ind = np.argsort(R11_R12, axis=1)
        alpha_bi_ = np.array([[res.params['alpha'].value, 1 - res.params['alpha'].value] for res in result_ajust_bi])
        alpha_bi = np.array([amp_bi_i[ind_i] for amp_bi_i, ind_i in zip(alpha_bi_, ind)])
        R11_R12 = np.sort(R11_R12, axis=1)

        # First plot: R1, R11, R12 VS B_relax
        ax = pdf.get_ax()
        ax.plot(B_relax, R1, '--d', c="darkviolet", label=r'$R_1$')
        ax.plot(B_relax, R11_R12.T[0], '--*', c='b', label=r'$R_1^{(1)}$')
        ax.plot(B_relax, R11_R12.T[1], '--*', c='#0081FE', label=r'$R_1^{(2)}$')
        ax.grid('on')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_title('Relaxation')

        # Second plot: alpha VS B_relax
        ax = pdf.get_ax()
        ax.plot(B_relax, alpha_bi.T[0], '--*', c='b', label=r'$a_{bi}$')
        ax.plot(B_relax, alpha_bi.T[1], '--*', c='#0081FE', label=r'($1-a_{bi}$)')
        ax.grid('on')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.set_xlabel(r"$B_{relax}$  [MHz]")
        ax.set_title('population')

        pdf.close_pdf()

        return R11_R12, alpha_bi


