from typing import List


class FitData:
    def __init__(self, data: np.ndarray, mask:np.ndarray, label:str):
        self.data = data
        self.mask = mask
        self.label = label


class BestFit:
    def __init__(self, data_matrix:List[FitData], color):
        self.data_matrix = data_matrix
        self.color = color

    def __getitem__(self, item):
        return self.data_matrix[item]