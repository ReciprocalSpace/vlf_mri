
__version__ = "1.0.0"
__author__ = "Aimé Labbé and Marie Poirier-Quinot"
__credit__ = "Université Paris-Saclay, CEA, CNRS, Inserm, BioMaps"
__all__ = ["import_vlf_file", "import_sdf_file", "FidData", "MagData", "RelData"]

from vlf_mri.code.data_reader import import_sdf_file, import_sdf2_file, import_vlf_file

from vlf_mri.code.fid_data import FidData
from vlf_mri.code.mag_data import MagData
from vlf_mri.code.rel_data import RelData

