import pickle
from abc import ABC
from datetime import date
from pathlib import Path


class VlfData(ABC):
    def __init__(self, data_file_path, data_type, best_fit):
        self.data_file_path = data_file_path
        self.experience_name = data_file_path.stem
        self.saving_folder = Path(data_file_path.parent) / ("result_" + self.experience_name)
        self.saving_vlf_folder = Path(self.saving_folder) / "vlf_data"
        if not self.saving_folder.is_dir():
            self.saving_folder.mkdir()
        if not self.saving_vlf_folder.is_dir():
            self.saving_vlf_folder.mkdir()

        self.data_type = data_type

        self.best_fit = best_fit

    def batch_plot(self, title):
        pass

    def report(self):
        pass

    def save_to_pdf(self, *args):
        pass

    def save_to_vlf(self, file_name=None):
        if file_name is None:
            file_name = f"{date.today().strftime('%Y%m%d')}-{self.data_type}-{self.experience_name}.vlf"
        file_path = self.saving_vlf_folder / file_name
        vlf_file = open(file_path, "wb")
        pickle.dump(self, vlf_file)
        vlf_file.close()
        return file_path

