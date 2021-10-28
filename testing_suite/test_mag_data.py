import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")
fid_data = vlf_mri.import_SDF_file(file)
fid_data.apply_mask(sigma=2, display_report=False)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)


def test_batch_plot():
    global file
    global mag_data_mean
    mag_data_mean.batch_plot("Test")


def test_to_rel():
    global file
    global mag_data_mean
    mag_data_mean.batch_plot("Test")
