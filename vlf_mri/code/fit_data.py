import numpy as np
from typing import List


class FitData:
    """
    Container for an element of best_fit dictionary attribute in the VlfData class

    This class contains the fit data for a given algorithm, and also allows some functionalities when plotting fit data,
    such as labeling curves, setting markers shape, size, and colors for instance. Please note that this formatting is
    common to every 1D array element in data. To vary the plot_keywords between curves, the FitDataArray class must be
    used in combination to the this class.

    Attributes
    ----------
    data : numpy.ndarray
        Fit data
    mask : numpy.ndarray of bools
        mask to apply on data
    plot_keywords : dict
        Keywords arguments to pass to the pyplot.plot method. If data is multidimensional, every 1D arrays of data will
        have the same plot formatting.
    """
    def __init__(self, data: np.ndarray, mask=None, **plot_keywords):
        """
        Initialize a FitData object

        Parameters
        ----------
        data : numpy.ndarray
            Fitted data. This array must be the same shape as the original data.
        mask : numpy.ndarray of bool
            Mask to apply on the fitted data.
        plot_keywords : dict
            Keywords arguments to pass when plotting the fitted data.
        """
        self.data = data
        self.mask = np.zeros_like(data, dtype=bool) if mask is None else mask
        self.plot_keywords = plot_keywords if plot_keywords is not None else {}

    def __getitem__(self, item):
        return FitData(self.data[item], self.mask[item], **self.plot_keywords)


class FitDataArray:
    """
    Implementation as array of FitData objects

    This class encapsulates a list of FitData objects and provides functionalities for indexation. This insures that the
    user does not have to know precisely how the dimensionality of the data is separated between this class and the
    FitData class. For instance, if the FID data is  d x n x m, then an instance of this class can contain a (d, m)
    array of FitData objects, each with size (n,), or a (d,) array of FitData objects, each with size (m, n).

    Attributes
    ----------
    fit_data_array : numpy.ndarray of FitData
        Array containing the the FitData objects
    """
    def __init__(self, fit_data_array: List[FitData]):
        self.fit_data_array = np.array(fit_data_array)

    def __len__(self):
        return len(self.fit_data_array)

    def __getitem__(self, item):
        output = self._recursive_getitem(self.fit_data_array, item)
        if isinstance(output, FitData):
            return output
        if isinstance(output, list):
            return FitDataArray(output)

    @staticmethod
    def _recursive_getitem(o, item):
        # Recursively returns the elements over the set of dimensions in item. This implementation allows to slice
        # through the FitDataArray and FitData objects without having to care for the object type.

        if isinstance(item, int) or isinstance(item, slice):
            return o[item]  # End of recursion
        else:  # item is a tuple
            o_0 = o[item[0]]
            item_1 = item[1:] if len(item) > 2 else item[1]
            if isinstance(item[0], int):
                return FitDataArray._recursive_getitem(o_0, item_1)
            else:
                return [FitDataArray._recursive_getitem(o_i, item_1) for o_i in o_0]
