import numpy as np

import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")

folder = Path("relax_test_data")
file = folder / "Sang_total_AC.sdf"

file = folder / "sang_total-2021-21-10_-18C-solide2.sdf"



# fid_data = vlf_mri.import_sdf_file(file)
fid_data = vlf_mri.import_sdf_file(file)
fid_data.batch_plot("sang_total-2021-21-10")
# fid_data.apply_mask(sigma=10, display_report=True)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)


def test_batch_plot():
    global mag_data_mean
    mag_data_mean.batch_plot("Test")


def test_apply_mask():
    global mag_data_mean
    mag_data_mean.apply_mask(sigma=10, display_report=True)


def test_to_string():
    global mag_data_mean
    print(mag_data_mean)


def test_slicing():
    global mag_data_mean
    # new_mag_data = mag_data_mean[1:3]
    # print(new_mag_data)

    rel_data = mag_data_mean.to_rel()
    new_mag_data = mag_data_mean[0:3, 5:]
    print(new_mag_data)
    new_mag_data.save_to_pdf()

    print(new_mag_data)


def test_to_rel():
    global mag_data_mean
    mag_data_mean.apply_mask(sigma=3, display_report=False)
    rel_data = mag_data_mean.to_rel()
    print(rel_data)


def test_save_to_pdf():
    global mag_data_mean
    mag_data_mean.apply_mask(sigma=3, display_report=False)
    rel_data = mag_data_mean.to_rel()
    # mag_data_mean.save_to_pdf(display=True)


def test_save_to_vlf():
    global mag_data_mean
    mag_data_mean.apply_mask(sigma=3, display_report=False)
    rel_data = mag_data_mean.to_rel()

    vlf_file_path = mag_data_mean.save_to_vlf()
    loaded_mag = vlf_mri.import_vlf_file(vlf_file_path)
    print(loaded_mag)


def test_to_ilt():
    global mag_data_mean
    lmd_folder = Path("ilt_lambda")
    for i, lmd in enumerate(np.logspace(-4, 3, 6*5+1)):
        ilt_data = mag_data_mean.to_ilt(lmd=lmd)
        ilt_data.batch_plot("sang_total-2021-21-10_-18C-solide2", lmd_folder / f"test_{i}.png")
    # mag_data_mean.batch_plot()

    # rel_data = mag_data_mean.to_rel()
    # rel_data.save_to_pdf(display=True)
    # mag_data_mean.save_to_pdf(display=True)


if __name__=="__main__":
    # test_batch_plot()
    # test_apply_mask()
    # test_slicing()
    # test_to_string()
    # test_to_rel()
    # test_save_to_pdf()
    # test_save_to_vlf()
    test_to_ilt()
