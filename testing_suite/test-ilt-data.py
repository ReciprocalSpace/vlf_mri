import copy

import vlf_mri
from pathlib import Path
import numpy as np
from scipy import optimize
from scipy import special
from lmfit import Model, Parameters
import matplotlib.pyplot as plt


def plot_r11(B_relax, R11):  # Affichage de courbes R1
    plt.figure(figsize=(8/2.54,8/2.54))
#     plt.loglog(B_relax, R11, "o-")
    plt.scatter(B_relax, R11, facecolors='none', edgecolors='b', linewidths=0.8, s = 20)
    plt.ylabel(r"$R_{11}$ [s$^{-1}$]")
    plt.xlabel(r"Frequence [MHz]")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([5e0, 5e3])
    plt.show()





def show_as_image(B_relax, R1, R11_1, ilt):
    xmin = B_relax.min()
    xmax = B_relax.max()
    ymin = R1[50:].min()
    ymax = R1.max()
    plt.imshow(np.rot90(ilt, -1)[50:], origin='lower', aspect=0.5)

    d = np.log(R11_1[::-1])
    # d = (d-d.min())/(d.max()-d.min())*len(ilt[50:].T)/3+38
    # plt.plot(d, c="w")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.axis("off")
    plt.show()


def load_fid_data_from_scratch() -> vlf_mri.FidData:
    folder = Path("relax_test_data")
    file = folder / "sang_total-2021-21-10_-18C-solide2.sdf"
    fid_data = vlf_mri.import_sdf_file(file)
    fid_data.apply_mask(5)
    return fid_data


def load_fid_data_from_vlf() -> vlf_mri.FidData:
    path = Path("relax_test_data\\result_sang_total-2021-21-10_-18C-solide2\\vlf_data\\20220531-FID-sang_total-2021-21"
                "-10_-18C-solide2.vlf")
    fid_data = vlf_mri.import_vlf_file(path)
    return fid_data


def load_ilt_data_from_vlf() -> vlf_mri.ILTData:
    path = Path("relax_test_data/result_sang_total-2021-21-10_-18C-solide2/vlf_data/20220531-ILT-sang_total-2021-21"
                "-10_-18C-solide2.vlf")
    ilt_data = vlf_mri.import_vlf_file(path)
    return ilt_data


def main():
    # fid_dat = load_fid_data_from_scratch()
    # fid_data = load_fid_data_from_vlf()
    # mag_data = fid_data.to_mag_mean()
    # R1 = np.logspace(-1, 4, 201)
    # ilt_data = mag_data.to_ilt(R1=R1, lmd=5e1, reg_order=2, penalty="c")

    ilt_data = load_ilt_data_from_vlf()
    ilt_data = ilt_data[::2, 20:180:5]
    # ilt_data.batch_plot("Sang_total_AC")

    # R11, alpha = ilt_data._get_r11(1, True)

    rel_data = ilt_data.to_rel(True)

    rel_data.save_to_pdf(display=True)

    # print(R11)
    # plt.plot(R1i[0])
    # plt.plot(R1i[1])
    # plt.show()

    # plot_r11(B_relax, R11_1)
    # show_as_image(B_relax, R1, R11_1, ilt)





if __name__ == "__main__":
    main()
