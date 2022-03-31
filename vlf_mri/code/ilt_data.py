import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cycler import cycler
from .vlf_data import VlfData


class ILTData(VlfData):
    def __init__(self, fid_file_path: Path, algorithm: str, alpha: np.ndarray, B_relax: np.ndarray,
                 R1: np.ndarray, lmd: np.ndarray, mask=None, best_fit=None, normalize=True):
        super().__init__(fid_file_path, "ILT", alpha, mask, best_fit)

        if normalize:
            self.data /= np.sum(self.data, axis=1, keepdims=True)  # Normalization

        self.B_relax = B_relax
        self.R1 = R1
        self.lmd = lmd

    def batch_plot(self, title=None, save=""):
        offset = 0.
        fig = plt.figure(figsize=(8/2.54, 10/2.54), tight_layout=True, dpi=150)
        ax = plt.gca()
        colormap = plt.cm.get_cmap("plasma")
        custom_colors = cycler('color', [colormap(x) for x in np.linspace(0, 0.9, len(self.B_relax))])
        ax.set_prop_cycle(cycler('color', custom_colors))

        for i, (R1_i, alpha_i, B_relax_i, lmd_i) in enumerate(zip(self.R1, self.data, self.B_relax, self.lmd)):
            ind0 = np.argmin(np.absolute(1 - R1_i))
            ind0 = 0

            plt.semilogx(R1_i[ind0:], R1_i[ind0:] ** 0 - 1 + offset, c="k", alpha=0.2, linewidth=1)
            plt.semilogx(R1_i[ind0:], alpha_i[ind0:]+offset)

            plt.title(title)

            plt.xlabel(r"Relaxation $R_1$ [s$^{-1}$]")
            plt.ylabel(r"Density $\alpha$ [arb. unit]")
            if i%3 == 0:
                plt.text(x=R1_i[ind0:].min(), y=offset + 0.01, s=rf"{B_relax_i:.1e}", fontsize=6)

            # plt.text(x=R1_i.min(), y=offset+0.01, s=rf"{B_relax_i:.1e} $\lambda$={lmd_i}", fontsize=8)
            offset += 0.05

        plt.ylim((-0.1, offset+0.1))
        plt.text(x=R1_i[ind0:].min(), y=offset+0.01, s=r"$B_{rel}$    $\lambda$="+f"{self.lmd[0]:.1e}", fontsize=8)
        if save:
            plt.savefig(save)
        plt.show()


    def to_rel(self):
        raise NotImplemented
