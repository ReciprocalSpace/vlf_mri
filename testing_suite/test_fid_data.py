import vlf_mri
from pathlib import Path


folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")


def test_import_SDF_file():
    print("*"*32+" test_import_SDF_file")
    fid_matrix = vlf_mri.import_sdf_file(file)
    print(type(fid_matrix))


def test_print_repr():
    print("*" * 32 + " test_print_repr" )
    fid_matrix = vlf_mri.import_sdf_file(file)
    print(repr(fid_matrix))
    print(fid_matrix)


def test_indexing():
    print("*" * 32 + " test_slice-1: 1D slice - multiple output")
    fid_matrix = vlf_mri.import_sdf_file(file)
    sliced_fid_matrix = fid_matrix[0:2]
    print(sliced_fid_matrix)

    print("*" * 32 + " test_slice-2: 1D slice - single output")
    fid_matrix = vlf_mri.import_sdf_file(file)
    sliced_fid_matrix = fid_matrix[0]
    print(sliced_fid_matrix)

    print("*" * 32 + " test_slice-3: 2D slice - multiple output")
    fid_matrix = vlf_mri.import_sdf_file(file)
    sliced_fid_matrix = fid_matrix[1:3, 4:5]
    print(sliced_fid_matrix)

    print("*" * 32 + " test_slice-3: 2D slice - multiple output")
    fid_matrix = vlf_mri.import_sdf_file(file)
    sliced_fid_matrix = fid_matrix[1:, 4]
    print(sliced_fid_matrix)


def test_apply_mask():
    print("*" * 32 + " test_apply_mask")
    fid_matrix = vlf_mri.import_sdf_file(file)
    fid_matrix.apply_mask(sigma=2, dims="xyz", display_report=True)
    print(fid_matrix)


def test_save_to_pdf():
    print("*" * 32 + " test_apply_mask")
    fid_matrix = vlf_mri.import_sdf_file(file)
    fid_matrix.apply_mask(sigma=2, dims="xyz", display_report=False)
    fid_matrix.save_to_pdf(fit_to_plot=None, display=False)


def test_fid_to_mag():
    fid_data = vlf_mri.import_sdf_file(file)
    fid_data.apply_mask(sigma=2, display_report=True)

    mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)
    mag_data_intercept = fid_data.to_mag_intercept()
    mag_data_mean.batch_plot("Mean")
    mag_data_intercept.batch_plot("Intercept")
    # mag_data_likelihood = fid_data.to_mag_max_likelihood()
    # mag_data_likelihood.batch_plot()
    fid_data.save_to_pdf(display=True)


def test_save_data_to_file():
    fid_data = vlf_mri.import_sdf_file(file)
    mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)

    pickle_file_path = fid_data.save_data_to_file()
    loaded_fid = vlf_mri.import_vlf_file(pickle_file_path)
    print(loaded_fid)


if __name__ == "__main__":
    # test_import_SDF_file()
    # test_print_repr()
    # test_indexing()
    # test_apply_mask()
    # test_save_to_pdf()
    # test_fid_to_mag()
    test_save_data_to_file()

