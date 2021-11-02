import numpy as np
from typing import List, Iterable

# TODO Complete this class
# TODO add field for plotting dictionaries
# TODO implement this in all VlfData child classes


class FitData:
    def __init__(self, data: np.ndarray, mask=None, **plot_keywords):
        self.data = data
        self.mask = np.zeros_like(data, dtype=bool) if mask is None else mask
        self.plot_keywords = plot_keywords if plot_keywords is not None else {}

    def __getitem__(self, item):
        return FitData(self.data[item], self.mask[item], **self.plot_keywords)


class FitDataArray:
    def __init__(self, fit_data_array: List[FitData]):
        self.fit_data_array = np.array(fit_data_array)

    def __len__(self):
        return len(self.fit_data_array)

    def __getitem__(self, item):
        output = self._recursive_get_item(self.fit_data_array, item)
        if isinstance(output, FitData):
            return output
        if isinstance(output, list):
            return FitDataArray(output)

    @staticmethod
    def _recursive_get_item(o, item):
        if isinstance(item, int) or isinstance(item, slice):
            return o[item]
        else:
            o_0 = o[item[0]]
            item_1 = item[1:] if len(item) > 2 else item[1]
            if isinstance(item[0], int):
                return FitDataArray._recursive_get_item(o_0, item_1)
            else:
                return [FitDataArray._recursive_get_item(o_i, item_1) for o_i in o_0]
