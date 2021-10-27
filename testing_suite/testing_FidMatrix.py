import vlf_mri
from pathlib import Path

folder = Path("relax_test_data")
file = Path("relax_test_data/sang-0p5C.sdf")


def test_import_SDF_file():
    print("*"*32+" test_import_SDF_file")
    global file
    fid_matrix = vlf_mri.import_SDF_file(file)
    print(type(fid_matrix))


def test_print_repr():
    print("*" * 32 + " test_print_repr" )
    fid_matrix = vlf_mri.import_SDF_file(file)
    print(repr(fid_matrix))
    print(fid_matrix)


def test_slice_1():
    print("*" * 32 + " test_slice-1: 1D slice - multiple output")
    fid_matrix = vlf_mri.import_SDF_file(file)
    sliced_fid_matrix = fid_matrix[0:2]
    print(sliced_fid_matrix)


def test_slice_2():
    print("*" * 32 + " test_slice-2: 1D slice - single output")
    fid_matrix = vlf_mri.import_SDF_file(file)
    sliced_fid_matrix = fid_matrix[0]
    print(sliced_fid_matrix)


def test_slice_3():
    print("*" * 32 + " test_slice-3: 2D slice - multiple output")
    fid_matrix = vlf_mri.import_SDF_file(file)
    sliced_fid_matrix = fid_matrix[1:3, 4:5]
    print(sliced_fid_matrix)


def test_slice_4():
    print("*" * 32 + " test_slice-3: 2D slice - multiple output")
    fid_matrix = vlf_mri.import_SDF_file(file)
    sliced_fid_matrix = fid_matrix[1:, 4]
    print(sliced_fid_matrix)


def test_apply_mask():
    print("*" * 32 + " test_apply_mask")
    fid_matrix = vlf_mri.import_SDF_file(file)
    fid_matrix.apply_mask(sigma=2, dims="xyz", display_report=True)

    print(fid_matrix)


if __name__ == "__main__":
    test_import_SDF_file()
    test_print_repr()
    test_slice_1()
    test_slice_2()
    test_slice_3()
    test_slice_4()
    test_apply_mask()
