import numpy as np
from abc import ABC
from pathlib import Path


class VlfData(ABC):
    def __init__(self, data_file_path, data_type, best_fit):
        self.sdf_file_path = data_file_path
        self.experience_name = data_file_path.stem
        self.saving_folder = Path(data_file_path.parent) / ("result_" + self.experience_name)
        if not self.saving_folder.is_dir():
            self.saving_folder.mkdir()

        self.data_type = data_type

        self.best_fit = best_fit

    def batch_plot(self, title):
        pass

    def report(self):
        pass

    def save_to_pdf(self, *args):
        pass
