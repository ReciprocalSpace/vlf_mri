import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang_total-2020-15-12_b.sdf")

file = Path("test_data.sdf")
fid_data = vlf_mri.import_sdf_file(file)
fid_data.batch_plot("Fid data with outliers!")
fid_data.apply_mask(sigma=5, display_report=True)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)
# mag_data_mean.apply_mask(sigma=3)
rel_data = mag_data_mean.to_rel()
mag_data_mean.save_to_pdf()


def test_str():
    global rel_data
    print(rel_data)


def test_save_to_pdf():
    global rel_data
    rel_data.save_to_pdf(display=True)


def test_save_to_vlf():
    global rel_data
    vlf_file_path = rel_data.save_to_vlf()
    loaded_rel = vlf_mri.import_vlf_file(vlf_file_path)
    print(loaded_rel)


if __name__ == "__main__":
    test_str()
    test_save_to_pdf()
    test_save_to_vlf()