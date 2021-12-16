import numpy as np

from vlf_mri import import_sdf_file, import_sdf2_file
from pathlib import Path


def test_import_SDF_file():
    folder = Path("relax_test_data")
    file = folder / "sang-0p5C.sdf"
    print(file)
    fid_matrix = import_sdf_file(file)
    print(fid_matrix)


if __name__ == "__main__":
    # test_import_SDF_file()
    folder = Path("relax_test_data")
    file = folder / "Sang_total_AC.sdf"
    fid_data = import_sdf2_file(file)

