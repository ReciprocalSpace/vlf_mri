import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")
fid_data = vlf_mri.import_SDF_file(file)
fid_data.apply_mask(sigma=2, display_report=False)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)


def test_batch_plot():
    global mag_data_mean
    mag_data_mean.batch_plot("Test")


def test_apply_mask():
    global mag_data_mean
    mag_data_mean.apply_mask(sigma=3, display_report=True)


def test_to_rel():
    global file
    global mag_data_mean
    mag_data_mean.batch_plot("Test")


if __name__=="__main__":
    # test_batch_plot()
    test_apply_mask()
