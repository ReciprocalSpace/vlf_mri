import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")
fid_data = vlf_mri.import_SDF_file(file)
fid_data.apply_mask(sigma=2, display_report=False)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)
mag_data_mean.apply_mask(sigma=3)
rel_data = mag_data_mean.to_rel()
mag_data_mean.save_to_pdf()


def test_str():
    global rel_data
    print(rel_data)


def test_save_to_pdf():
    global rel_data
    rel_data.save_to_pdf(display=True)


if __name__ == "__main__":
    test_str()
    test_save_to_pdf()
