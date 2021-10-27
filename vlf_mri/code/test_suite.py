#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:52:07 2021

@author: poquima
"""

import matplotlib as mpl
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from lmfit import Model, Parameters, report_fit
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ma
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rice
from ffc_analyze_methods import import_SDF_file, apply_mask, PDFSaver, save_fid_to_pdf
from ffc_analyze_methods import get_normalized_magnetization  # FID -> MAG General case
from ffc_analyze_methods import mean_magnetization, intersect_magnetization, max_likelihood  # FID -> Mag algorithms
from ffc_analyze_methods import get_relaxation_times, plot_relaxation, plot_magnetization
from ffc_analyze_methods import analyse_single_sdf_file
from ffc_analyze_methods import analyse_all_sdf_file_in_directory

mpl.rc('font', **{'size': 9})


class TestingSuite:
    @staticmethod
    def test_library():
        folder_manip = Path("sample_data")
        sdf_file = folder_manip / "sang-0p5C.sdf"
        name_manip = "sang-0p5C"

        # Lecture et maskage des données aberrantes
        B_relax, tau, t_fid, fid_matrix = import_SDF_file(sdf_file, False)
        fid_matrix = apply_mask(fid_matrix, 3, False, B_relax, tau, t_fid)

        # Calcul des aimantations: 3 approches différentes
        magnetization_mean, best_fit_mean = get_normalized_magnetization(mean_magnetization, fid_matrix, 5, 150)
        # magnetization_intersect, best_fit_intersect = get_normalized_magnetization(intersect_magnetization, fid_matrix, 5, 150)
        # magnetization_maxlikelihood, best_fit_maxlikelihood = get_normalized_magnetization(max_likelihood, fid_matrix)

        # Enregistrement des FID en pdf avec les courbes pour l'évaluation des aimantations
        # save_fid_to_pdf(B_relax[0], tau[0], t_fid, fid_matrix[0], best_fit_mean[0], "sang-0p5C-mean", folder_manip)
        # save_fid_to_pdf(B_relax[0], tau[0], t_fid, fid_matrix[0], best_fit_intersect[0], "sang-0p5C-intersect", folder_manip)
        # save_fid_to_pdf(B_relax[0], tau[0], t_fid, fid_matrix[0], best_fit_maxlikelihood[0], "sang-0p5C-likelihood",
        #                 folder_manip)

        # Calcul des temps de relaxation

        result_mono, result_bi = get_relaxation_times(tau, magnetization_mean)

        plot_magnetization(tau, magnetization_mean, B_relax, result_mono, result_bi, name_manip, folder_manip, True)
        plot_relaxation(result_mono, result_bi, B_relax, name_manip, folder_manip)

    @staticmethod
    def test_analyse_single_sdf_file():

        file_path = Path("sample_data/sang-0p5C.sdf")
        save_folder = Path("sample_data/")
        analyse_single_sdf_file(file_path)


if __name__ == '__main__':
    testing_suite = TestingSuite
    # testing_suite.test_library()
    testing_suite.test_analyse_single_sdf_file()
    #
    # testing_suite.test_pdf_saver()
