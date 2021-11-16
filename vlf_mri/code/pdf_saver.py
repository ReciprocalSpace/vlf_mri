import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# TODO : add kwds argument for more flexibility with page layout
class PDFSaver:
    """
    Save multiple plots to a pdf file

    This class saves a list of plots to a single pdf file. Each page contains n x m axes positioned on a regular lattice.
    The total number of axes does not need to be known in advance. Instead, the user can ask for a new axis as he needs
    them using the get_ax() method. If a page is full, then the page is stored and a new one is created with empty axes.

    This class manages the operation on the pdf file.

    Attributes
    ----------
    pdf_file: matplotlib.backends.backend_pdf.PdfPages
        pdf file where the pdf pages are saved
    n_rows : int
        Number of rows of axes per page.
     n_columns : int
        Number of columns of axes per page.
    title : str
        Title of the document (appears on each page). Default is an empty string.
    current_page : PdfPage
        Active page in the Pdf file
    flag_display : bool
        Specify whether to display each pdf page to the user as a figure in the console or not.

    Methods
    -------
    get_ax():
        Returns a new ax object
    close_pdf():
        Close the pdf file

    Examples
    --------
    # The following code assumes pdf_saver is at the root directory of the python script. It generates a pdf file
    # containing two pages with 12 axes (3 x 4), where the second page is half filled.

    >>> from pdf_saver import PDFSaver
    >>> import numpy as np
    >>> pdf = PDFSaver("my_pdf.pdf", 3, 4, "Hello world!", True)
    >>> for i in range(16):
    >>>    ax = pdf.get_ax()
    >>>    x = np.linspace(0., 1., 51)
    >>>    y = np.sin(x * 2 * np.pi * np.random.randn() + 2 * np.pi * np.random.rand())
    >>>    ax.plot(x, y)
    >>> pdf.close_pdf()
    """
    def __init__(self, file_path: Path, n_columns: int, n_rows: int, title="", display_pages=False) -> None:
        """
        Create a Pdf file where plots can be saved.

        Create a Pdf file where each page contains n_columns x n_roms axes and a general title.

        Parameters
        ----------
        file_path : Path
            Path where the Pdf is saved
        n_columns : int
            Number of columns of axes per page.
        n_rows : int
            Number of rows of axes per page.
        title : str
            Title of the document (appears on each page). Default is an empty string.
        display_pages : bool
            Specify whether to display each pdf page to the user as a figure in the console or not.
        """
        self.pdf_file = PdfPages(file_path)
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.title = title

        self.current_page = PDFPage(self.n_rows, self.n_columns, title)
        self.flag_display = display_pages

    def get_ax(self) -> Axes:
        """
        Return a new axis

        Return a new axis handle to the user when needed. If the current pdf page is full, the page is stored, and this
        method returns an axis on a fresh page.

        Returns
        -------
        axis handle
        """
        if self.current_page.flag_end_of_page:
            self._store_page()
            self.current_page = PDFPage(self.n_rows, self.n_columns, self.title)
        ax = self.current_page.get_new_ax_handle()
        return ax

    def close_pdf(self) -> None:
        """
        Close the Pdf file

        Process all empty axes on the last page and close the pdf file.

        Returns
        -------

        """
        self._store_page()
        self.pdf_file.close()

    def _store_page(self) -> None:
        """
        Store the active pdf page to the pdf file

        Process all empty axes on the page and store the pdf page in the pdf file. Also display the figure to the user
        if the flag_display is True.

        Returns
        -------
        """
        fig = self.current_page.flush()
        self.pdf_file.savefig(fig)
        if self.flag_display:
            plt.show()
        else:
            plt.close()


class PDFPage:
    """
    Manage a single pdf page or figure object

    This class initializes a figure object with n x m  axes and provides axes to the user when requested. This class
    works in combination with the PdfSaver class and should not be directly accessed by the user.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Figure handle
    axes : matplotlib.figure.Figure
        List of all axes handles in the figure
    _it : int
        Index of the next axis handle to provide to the user
    flag_end_of_page : bool
        Signal that is True when all axes have been send to the user

    Methods
    -------
    get_new_ax_handle():
        Provide the next unused axis handle
    flush():
        Set all unused axes in figure to "off"
    """
    def __init__(self, n_columns:int,  n_rows: int, title="") -> None:
        """
        Initialize a PdfPage object

        Instantiate a figure with n_columns x n_rows axes with a sup title.

        Parameters
        ----------
        n_columns : int
            Number of columns of axes per page.
        n_rows : int
            Number of rows of axes per page.
        title : str, optional
            Figure sup title. Default is an empty string.
        """
        self.format = format
        fig, axes = plt.subplots(n_columns, n_rows, figsize=(21 / 2.54 - 2, 29 / 2.54 - 2), dpi=120, tight_layout=True)
        fig.suptitle(title)
        self.fig = fig
        self.axes = np.array(axes).flatten()
        self._it = 0
        self.flag_end_of_page = False

    def get_new_ax_handle(self) -> Axes:
        """
        Provide the next unused axis handle

        This methods provides the next axis handle to the user when called. When no more empty ax is available, the
        flag_end_of_file signal is set to True.

        Returns
        -------
        axis handle
        """
        ax = self.axes[self._it]
        self._it += 1
        if self._it >= len(self.axes):
            self.flag_end_of_page = True
        return ax

    def flush(self) -> Figure:
        """
        Set all unused axes in figure to "off"

        This method sets all remaining axes to "off" so that the final pdf file does not contain empty axes. Call this
        method when finishing a pdf page.

        Returns
        -------
        figure handle

        """
        while not self.flag_end_of_page:
            ax = self.get_new_ax_handle()
            ax.axis('off')
        return self.fig
