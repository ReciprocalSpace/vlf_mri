#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:52:07 2021

@author: poquima
"""

# import glob
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from lmfit import Model, Parameters, report_fit
# from math import ceil
#
# from numpy import ma
# from pathlib import Path
# from scipy.optimize import minimize
#
# mpl.rc('font', **{'size': 9})


# def analyse_all_sdf_file_in_directory(directory):
#     directory = Path(directory)
#     sdf_file_path_generator = glob.glob(directory / '*.sdf')
#
#     for sdf_file_path in sdf_file_path_generator:
#         analyse_single_sdf_file(sdf_file_path)
#

# def analyse_single_sdf_file(sdf_file_path: Path, mask_filter_criterion=3.):
#     experience_name = sdf_file_path.stem  # filename without extension
#     experience_folder = sdf_file_path.parent  # folder
#
#     save_folder = Path(experience_folder) / ("result_" + experience_name)
#     if not save_folder.is_dir():
#         save_folder.mkdir()
#
#     # FID_matrix
#     B_relax, tau, t_fid, fid_matrix = import_SDF_file(sdf_file_path, True)
#     fid_matrix = apply_mask(fid_matrix, mask_filter_criterion, True, B_relax, tau, t_fid)
#
#     # MAGNETIZATION
#     magnetization_mean, best_fit_fid_mean = get_normalized_magnetization(mean_magnetization, fid_matrix, 4, 128)
#
#     for B_i, tau_i, fid_matrix_i, best_fit_fid_i in zip(B_relax, tau, fid_matrix, best_fit_fid_mean):
#         save_fid_to_pdf(B_i, tau_i, t_fid, fid_matrix_i, best_fit_fid_i, experience_name, save_folder)
#
#     # RELAXATION TIMES
#     result_mono, result_bi = get_relaxation_times(tau, magnetization_mean)
#
#     plot_magnetization(tau, magnetization_mean, B_relax, result_mono, result_bi, experience_name, save_folder, True)
#     plot_relaxation(result_mono, result_bi, B_relax, experience_name, save_folder)
