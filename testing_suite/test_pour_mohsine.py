import vlf_mri
from pathlib import Path

if __name__ == "__main__":
    folder = Path("relax_test_data")
    # file = Path("relax_test_data/Sang_total_surnageant_corrige_AL.sdf")

    file = folder / "Sang_total_AC.sdf"
    fid_data = vlf_mri.import_sdf2_file(file)

    # fid_data.batch_plot()

    # print(fid_data.data.shape)

    # path = fid_data.save_to_vlf()

    # fid_data.apply_mask(sigma=2, display_report=True)
    # mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)

    # mag_data_mean.apply_mask(sigma=3)
    # rel_data = mag_data_mean.to_rel()
    # mag_data_mean.save_to_pdf()
