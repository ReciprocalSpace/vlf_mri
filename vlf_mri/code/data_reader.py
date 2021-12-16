import numpy as np
import pickle

from pathlib import Path
from typing import Union

from vlf_mri.code.fid_data import FidData
from vlf_mri.code.mag_data import MagData
from vlf_mri.code.rel_data import RelData

# TODO : refactor using factory design pattern


def import_sdf_file(sdf_file_path: Path) -> FidData:
    """
    Read and import an SDF file

    Read and import an SDF file and output a FidData object.

    Parameters
    ----------
    sdf_file_path : Path
        Path to the *.sdf file containing the relaxometry data.

    Returns
    -------
    fid_data : FidData
        object containing the FID data. See vlf_mri.FidData for more information.

    """
    print(f"Importing datafile: {sdf_file_path}")
    with open(sdf_file_path, 'r') as datafile:
        list_of_lines_in_sdf_file = datafile.readlines()

    # initialize some variable as lists:
    T1MX, B_relax, tau_init, tau_end, len_tau, raw_data, flag_log = [], [], [], [], [], [], []
    len_fid, delta_t_fid = [], []
    save_data = {"T1MX=": lambda _words: T1MX.append(float(_words[1])),
                 "BRLX=": lambda _words: B_relax.append(float(_words[1])),
                 "BS": lambda _words: len_fid.append(int(_words[2])),
                 "DW": lambda _words: delta_t_fid.append(float(_words[2])),
                 "NBLK=": lambda _words: len_tau.append(int(_words[1])),
                 "BINI=": lambda _words: tau_init.append(
                     float(4 * T1MX[-1] if _words[1] == "(4*T1MX)" else _words[1])),
                 "BEND=": lambda _words: tau_end.append(
                     float(0.01 * T1MX[-1] if _words[1] == "(0.01*T1MX)" else _words[1])),
                 "BGRD=": lambda _words: flag_log.append(_words[1] == 'LOG')
                 }

    # arrangement des listes
    flag_data = False
    for line in list_of_lines_in_sdf_file:
        words = line.split()
        if len(words) != 0:
            if flag_data:
                raw_data.append(float(words[2]))
            elif words[0] == r'DATA=':
                flag_data = True
            elif words[0] in save_data:
                save_data[words[0]](words)  # Store data in the relevant list
        else:
            flag_data = False

    # Format outputs: B_relax, tau, t_fid, data
    B_relax = np.array(B_relax)
    if flag_log[0]:
        tau = np.array([tau_0 * np.logspace(0., 1., len_tau[0], base=tau_1 / tau_0)
                        for tau_0, tau_1 in zip(tau_init, tau_end)])
    else:
        tau = np.array([np.linspace(tau_0, tau_1, len_tau[0]) for tau_0, tau_1 in zip(tau_init, tau_end)])

    t_fid = np.linspace(0, len_fid[0] * delta_t_fid[0], len_fid[0])
    fid_matrix_shape = (len(B_relax), len_tau[0], len_fid[0])
    fid_matrix = np.array(raw_data).reshape(fid_matrix_shape)

    return FidData(sdf_file_path, fid_matrix, B_relax, tau, t_fid)


def import_sdf2_file(sdf_file_path: Path) -> FidData:
    sdf_file_importer = ImportSdfFileV2()
    return sdf_file_importer(sdf_file_path)


class ImportSdfFileV2:
    def __init__(self):
        self.fid = []
        self.tau = []
        self.B_rel = []

        self.len_fid = None
        self.len_tau = None
        self.T1MAX = None
        self.delta_t = None
        self.tau_equation = None
        self.state = self.file_header

        self.FLAG_END_OF_FILE = False

    def __call__(self, sdf2_file_path: Path):
        self.__init__()

        with open(sdf2_file_path, "r") as file:
            while not self.FLAG_END_OF_FILE:
                self.state(file)
                self.state = self.find_state(file)

        self.fid = np.array(self.fid)
        self.tau = np.array(self.tau)
        self.B_rel = np.array(self.B_rel)

        t_fid = np.linspace(0, self.delta_t*self.len_fid, self.len_fid)

        return FidData(sdf2_file_path,self.fid,self.B_rel,self.tau,t_fid)

    def find_state(self, file):
        line = file.readline().rstrip()
        # print("\tFind state:\t",  line)
        word = line.split(" ")[0]
        if word == "NMRD":
            return self.skip_line
        if word == "PARAMETER":
            return self.parameter_summary
        if word == "ZONE":
            return self.zone_header
        if word == "DATA":
            return self.data
        if word == "":
            self.FLAG_END_OF_FILE = True
            return None
        # print("NOT FOUND:\t", repr(line))
        raise NotImplementedError

    def file_header(self, file):
        # print("Current state:\tFile header")
        for _ in range(10):
            _ = file.readline()

    def skip_line(self, file):
        # print("Current state:\tParameter summary")
        _ = file.readline().rstrip()

    def parameter_summary(self, file):
        # print("Current state:\tParameter summary")
        line = file.readline().rstrip()
        while line:
            words = line.split(" = ")
            key, value = words[0], words[1]

            if key == "BS":
                self.len_fid = int(float(value))
            elif key == "TAU":
                # ex: words[1] = [log:4*T1MAX:0.01*T1MAX:32] (equation)
                t = value[1:-1].split(":")  # On retire les "[" et "]"

                tau_min = t[1][:-5] + "self.T1MAX"
                tau_max = t[2][:-5] + "self.T1MAX"
                len_tau = t[3]

                self.len_tau = int(t[3])
                if t[0] == "log":
                    # tau_min * np.logspace(0., 1., len_tau[0], base=tau_1 / tau_0)
                    self.tau_equation = (
                        f"x={tau_min}*np.logspace(0, 1, {len_tau}, base=({tau_max})/({tau_min}))"
                    )
                elif t[0] == "lin":
                    self.tau_equation = (
                        f"x=np.linspace({tau_min}, {tau_max}, {len_tau})"
                    )
            elif key == "DW":
                self.delta_t = float(value)
            line = file.readline().rstrip()

    def zone_header(self, file):
        # print("Current state:\tZone header")
        line = file.readline().rstrip()
        while line:
            key, value = tuple(line.split(" = "))
            if key == "BR":
                self.B_rel.append(float(value))
            elif key == "T1MAX":
                self.T1MAX = float(value)
                # print(self.tau_equation)
                out = {}
                exec(self.tau_equation, {'self': self, 'np':np}, out)
                self.tau.append(out["x"])
            line = file.readline().rstrip()

    def data(self, file):
        # print("Current state:\tData")
        _ = file.readline()
        line = file.readline().rstrip()
        data = []
        while line:
            data.append(float(line.split("\t\t")[2]))
            line = file.readline().rstrip()
        data = np.array(data).reshape(self.len_tau, self.len_fid)
        self.fid.append(data)


def import_vlf_file(vlf_file_path: Path) -> Union[FidData, MagData, RelData]:
    """
    Import a vlf file

    Read a *.vlf file and return a VlfData object, which is either a FidData, MagData or a RelData object. The
    returned object is updated to the newest version of the library. Alternatively, the *.vlf file can be imported
    using the pickle package.

    Parameters
    ----------
    vlf_file_path : Path
        Path to the *.vlf binary file.

    Returns
    -------
    vlf_data : FidData, MagData, or RelData
        Data object saved in the *.vlf file. The object is updated to the newest library version.
    """
    vlf_file = open(vlf_file_path,  "rb")
    old_vlf_object = pickle.load(vlf_file)
    vlf_file.close()

    # Update object to a newer version of VlfData class
    data_file_path = old_vlf_object.data_file_path
    data_type = old_vlf_object.data_type
    mask = old_vlf_object.mask
    best_fit = old_vlf_object.best_fit

    vlf_data = old_vlf_object.data
    B_relax = old_vlf_object.B_relax
    if data_type == "REL":
        return RelData(data_file_path, vlf_data, B_relax, mask, best_fit)

    tau = old_vlf_object.tau
    if data_type == "MAG":
        algo = old_vlf_object.algorithm
        return MagData(data_file_path, algo, vlf_data, B_relax, tau, mask, best_fit)

    t_fid = old_vlf_object.t_fid
    if data_type == "FID":
        fid_matrix = old_vlf_object.data
        return FidData(data_file_path, fid_matrix, B_relax, tau, t_fid, mask, best_fit)
