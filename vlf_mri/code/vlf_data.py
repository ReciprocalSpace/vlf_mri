from __future__ import annotations

import numpy as np
import pickle

from abc import ABC
from datetime import date
from pathlib import Path


class VlfData(ABC):
    """
    Generic data class for the vlf_mri data

    This class contains the attributes and methods common FidData, MagData, and RelData, and manages
    data exports.

    Attributes
    ----------
    data_file_path : Path
        Path to the original *.sdf file containing the experimental data
    experience_name : str
        Name of the current experiment. The name is extracted from the file name. For example, if the file name is
        "my-experiment.sdf", then this field would be "my-experiment"
    saving_folder : Path
        Path to the folder where the file outputs (namely, pdf files) will be saved. By default, this field is
        constructed using the experience_name field. For example, if experience_name is "my-experiment", then this field
        would be "/result_my-experiment/"
    saving_vlf_folder : Path
        Path to the folder where the *.vlf files are saved. Vlf files are binary files containing a VlfData object
        (see save_to_vlf() method for more information). This folder is a sub directory of the saving_folder directory.
        For example, if example_name is "my-experiment", then this field is "/result_my-experiment/vlf_data/".

    Methods
    -------
    update_mask(new_mask, mode):
        Update the object mask on data
    save_to_vlf(file_name) -> Path:
        Save the object to a *.vlf file

    """
    def __init__(self, data_file_path, data_type, vlf_data, mask, best_fit):
        if mask is None:
            mask = np.zeros_like(vlf_data, dtype=bool)

        if best_fit is None:
            best_fit = {}

        self.data_file_path = data_file_path
        self.experience_name = data_file_path.stem
        self.saving_folder = Path(data_file_path.parent) / ("result_" + self.experience_name)
        self.saving_vlf_folder = Path(self.saving_folder) / "vlf_data"

        if not self.saving_folder.is_dir():
            self.saving_folder.mkdir()
        if not self.saving_vlf_folder.is_dir():
            self.saving_vlf_folder.mkdir()

        self.data_type = data_type
        self.data = vlf_data
        self.mask = mask
        self.best_fit = best_fit

    def update_mask(self, new_mask, mode="merge") -> VlfData:
        """Update the object mask on data
        Update the old data mask with the new mask. Depending on the value of value of the mode keyword, the mask can
        be merged with the old one (logical or) or replace it.


        Parameters
        ----------
        new_mask : numpy.ndarray of bool
            New mask

        mode : str
            "merge" or "replace" the new mask with the old mask. Default: "merge".

        Returns
        -------
        FidData, MagData or VlfData

        """
        if mode == "merge":
            self.mask = np.logical_or(self.mask, new_mask)
        elif mode == "replace":
            self.mask = new_mask
        return self

    def save_to_vlf(self, file_name=None) -> Path:
        """Save the object to a *.vlf file

        Export the object into a *.vlf binary file using the pickle package. If no filename is probided, the file is
        saved in the experience saving folder and is name by default "YYMMDD-<data type>-<experience name>.vlf" where
        <data type> = FID, MAG or REL.

        The object can then be reloaded using the vlf_mri.import_vlf_file(vlf_file_path) method, or alternatively using
        the pickle package (see the package documentation for more information.

        This method returns the *.vlf file path.

        Parameters
        ----------
        file_name : str, optional
            file_name of the binary file containing the object. The extension does not have to be *.vlf

        Returns
        -------
        vlf_file_path: Path
            Path to the *.vlf data file

        """
        if file_name is None:
            file_name = f"{date.today().strftime('%Y%m%d')}-{self.data_type}-{self.experience_name}.vlf"
        file_path = self.saving_vlf_folder / file_name
        vlf_file = open(file_path, "wb")
        pickle.dump(self, vlf_file)
        vlf_file.close()
        return file_path

    def batch_plot(self, title):
        """
        Produce a figure from the vlf data object

        Parameters
        ----------
        title : str
            Title of figure

        Raises
        ------
        NotImplemented
        """
        raise NotImplemented

    def report(self):
        """
        Print a report of the information in the vlf object

        Raises
        ------
        NotImplemented
        """
        raise NotImplemented

    def save_to_pdf(self, *args):
        """
        Create a pdf out of the data in the object

        Parameters
        ----------
        *args
            Arguments specific to the child class

        Raises
        ------
        NotImplemented
        """
        raise NotImplemented
