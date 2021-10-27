import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# TODO : add kwds argument for more flexibility with page layout
class PDFSaver:
    def __init__(self, file_path, n_columns, n_rows, title="", display_pages=False, ):
        self.pdf = PdfPages(file_path)
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.title = title

        self.current_page = PDFPage(self.n_rows, self.n_columns, title)
        self.flag_display = display_pages

    def get_ax(self):
        if self.current_page.flag_end_of_page:
            self._store_page()
            self.current_page = PDFPage(self.n_rows, self.n_columns, self.title)
        ax = self.current_page.get_new_ax_handle()
        return ax

    def close_pdf(self):
        self._store_page()
        self.pdf.close()

    def _store_page(self):
        fig = self.current_page.flush()
        self.pdf.savefig(fig)
        if self.flag_display:
            plt.show()
        else:
            plt.close()


class PDFPage:
    def __init__(self, n_columns,  n_rows, title=""):
        self.format = format
        fig, axes = plt.subplots(n_columns, n_rows, figsize=(21 / 2.54 - 2, 29 / 2.54 - 2), dpi=120, tight_layout=True)
        fig.suptitle(title)
        self.fig = fig
        self.axes = np.array(axes).flatten()
        self._it = 0
        self.flag_end_of_page = False

    def get_new_ax_handle(self):
        ax = self.axes[self._it]
        self._it += 1
        if self._it >= len(self.axes):
            self.flag_end_of_page = True
        return ax

    def flush(self):
        while not self.flag_end_of_page:
            ax = self.get_new_ax_handle()
            ax.axis('off')
        return self.fig
