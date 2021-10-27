#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:52:07 2021

@author: poquima
"""

import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from lmfit import Model, Parameters, report_fit
from math import ceil

from numpy import ma
from pathlib import Path
from scipy.optimize import minimize

mpl.rc('font', **{'size': 9})


def plot_relaxation(result_ajust_mono, result_ajust_bi, B_relax, name_manip, folder):
    # Create the pdfsaver object
    file_name = f"{name_manip}_Relaxation.pdf"
    file_path = os.path.join(folder, file_name)
    title = f"{name_manip} - Magnetization"
    pdf = PDFSaver(file_path, 1, 3, title, True)

    # Prepare and sort the R1, R11, R12 and alpha data into arrays
    R1 = np.array([res.params['R1'].value for res in result_ajust_mono])
    R11_R12 = np.array([[res.params['R11'].value, res.params['R12'].value] for res in result_ajust_bi])
    ind = np.argsort(R11_R12, axis=1)
    alpha_bi_ = np.array([[res.params['alpha'].value, 1 - res.params['alpha'].value] for res in result_ajust_bi])
    alpha_bi = np.array([amp_bi_i[ind_i] for amp_bi_i, ind_i in zip(alpha_bi_, ind)])
    R11_R12 = np.sort(R11_R12, axis=1)

    # First plot: R1, R11, R12 VS B_relax
    ax = pdf.get_ax()
    ax.plot(B_relax, R1, '--d', c="darkviolet", label=r'$R_1$')
    ax.plot(B_relax, R11_R12.T[0], '--*', c='b', label=r'$R_1^{(1)}$')
    ax.plot(B_relax, R11_R12.T[1], '--*', c='#0081FE', label=r'$R_1^{(2)}$')
    ax.grid('on')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_title('Relaxation')

    # Second plot: alpha VS B_relax
    ax = pdf.get_ax()
    ax.plot(B_relax, alpha_bi.T[0], '--*', c='b', label=r'$a_{bi}$')
    ax.plot(B_relax, alpha_bi.T[1], '--*', c='#0081FE', label=r'($1-a_{bi}$)')
    ax.grid('on')
    ax.legend(loc='best')
    ax.set_xscale('log')
    ax.set_xlabel(r"$B_{relax}$  [MHz]")
    ax.set_title('population')

    pdf.close_pdf()

    return R11_R12, alpha_bi


def analyse_all_sdf_file_in_directory(directory):
    directory = Path(directory)
    sdf_file_path_generator = glob.glob(directory / '*.sdf')

    for sdf_file_path in sdf_file_path_generator:
        analyse_single_sdf_file(sdf_file_path)


def analyse_single_sdf_file(sdf_file_path: Path, mask_filter_criterion=3.):
    experience_name = sdf_file_path.stem  # filename without extension
    experience_folder = sdf_file_path.parent  # folder

    save_folder = Path(experience_folder) / ("result_" + experience_name)
    if not save_folder.is_dir():
        save_folder.mkdir()

    # FID_matrix
    B_relax, tau, t_fid, fid_matrix = import_SDF_file(sdf_file_path, True)
    fid_matrix = apply_mask(fid_matrix, mask_filter_criterion, True, B_relax, tau, t_fid)

    # MAGNETIZATION
    magnetization_mean, best_fit_fid_mean = get_normalized_magnetization(mean_magnetization, fid_matrix, 4, 128)

    for B_i, tau_i, fid_matrix_i, best_fit_fid_i in zip(B_relax, tau, fid_matrix, best_fit_fid_mean):
        save_fid_to_pdf(B_i, tau_i, t_fid, fid_matrix_i, best_fit_fid_i, experience_name, save_folder)

    # RELAXATION TIMES
    result_mono, result_bi = get_relaxation_times(tau, magnetization_mean)

    plot_magnetization(tau, magnetization_mean, B_relax, result_mono, result_bi, experience_name, save_folder, True)
    plot_relaxation(result_mono, result_bi, B_relax, experience_name, save_folder)


def main():
    FilesOI = []
    DirOI = Path("sample_data/")
    for (_, _, filenames) in os.walk(DirOI):  # variable "_" ne sera jamais utilisée.
        files = filenames
        for f in files:
            if f[-3:] == 'sdf':
                FilesOI.append(f)

    for filename in FilesOI:
        name_manip = os.path.splitext(filename)[0]  # Nom du fichier sans extension
        folder = DirOI / name_manip

        if not os.path.exists(folder):
            os.makedirs(folder)

        B_relax, tau, t_fid, fid_matrix = import_SDF_file(DirOI / filename)


        # mag, best_fit_fid = get_normalized_magnetization(mean_magnetization, fid_matrix, 4, 128)
        mag, best_fit_fid = get_normalized_magnetization(max_likelihood, fid_matrix)

        for raw_data_i, best_fit_i, B_relax_i in zip(fid_matrix, best_fit_fid, B_relax):
            save_fid_to_pdf(raw_data_i, best_fit_i, t_fid, tau, B_relax_i, name_manip + '-likelihood', folder)

        result_ajust_mono, result_ajust_bi = get_relaxation_times(tau, mag)

        # plot_magnetization(tau, mag, B_relax, result_ajust_mono, result_ajust_bi,
        #                  name_manip, folder, display_residu=True)
        # plot_relaxation(result_ajust_mono, result_ajust_bi, B_relax, name_manip, folder)

        plot_magnetization(tau, mag, B_relax, result_ajust_mono, result_ajust_bi, name_manip + '-likelihood', folder,
                           display_residu=True)
        plot_relaxation(result_ajust_mono, result_ajust_bi, B_relax, name_manip + '-likelihood', folder)


if __name__ == '__main__':
    main()
