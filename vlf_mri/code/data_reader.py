import numpy as np
import pickle

from pathlib import Path

import vlf_mri


def import_sdf_file(sdf_file_path: Path):
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

    return vlf_mri.FidData(sdf_file_path, fid_matrix, B_relax, tau, t_fid)


def import_vlf_file(vlf_file_path: Path):
    vlf_file = open(vlf_file_path,  "rb")
    old_vlf_object = pickle.load(vlf_file)
    vlf_file.close()

    # Update object to a newer version of import VlfData class
    data_file_path = old_vlf_object.data_file_path
    data_type = old_vlf_object.data_type
    mask = old_vlf_object.mask
    best_fit = old_vlf_object.best_fit

    vlf_data = old_vlf_object.data
    B_relax = old_vlf_object.B_relax
    if data_type == "REL":
        return vlf_mri.RelData(data_file_path, vlf_data, B_relax, mask, best_fit)

    tau = old_vlf_object.tau
    if data_type == "MAG":
        algo = old_vlf_object.algorithm
        return vlf_mri.MagData(data_file_path, algo, vlf_data, B_relax, tau, mask, best_fit)

    t_fid = old_vlf_object.t_fid
    if data_type == "FID":
        fid_matrix = old_vlf_object.data
        return vlf_mri.FidData(data_file_path, fid_matrix, B_relax, tau, t_fid, mask, best_fit)
