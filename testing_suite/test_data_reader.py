from vlf_mri import import_SDF_file
from pathlib import Path


def test_import_SDF_file():
    file = Path("relax_test_data/sang-0p5C.sdf")
    print(file)
    fid_matrix = import_SDF_file(file)
    print(fid_matrix)


if __name__ == "__main__":
    test_import_SDF_file()

