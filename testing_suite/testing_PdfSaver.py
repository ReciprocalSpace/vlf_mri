from vlf_mri import PDFSaver
from pathlib import Path
import numpy as np

folder = Path("pdfsaver_output")


def _plot_random(ax):
    x = np.linspace(0., 1., 51)
    y = np.sin(x * 2 * np.pi * np.random.randn() + 2 * np.pi * np.random.rand())
    ax.plot(x, y)


def test_one_ax():
    global folder
    filename = folder / "test_one_ax.pdf"
    pdf = PDFSaver(filename, 3, 4, "", True)
    ax = pdf.get_ax()
    _plot_random(ax)
    pdf.close_pdf()


def test_one_complete_page():
    global folder
    filename = folder / "test_one_complete_page.pdf"
    pdf = PDFSaver(filename, 3, 4, "One full page", True)
    for i in range(12):
        ax = pdf.get_ax()
        _plot_random(ax)
    pdf.close_pdf()


def test_general_case():
    global folder
    filename = folder / "test_general_case.pdf"
    pdf = PDFSaver(filename, 3, 4, "General case", True)
    for i in range(16):
        ax = pdf.get_ax()
        _plot_random(ax)
    pdf.close_pdf()


def test_option_dont_display_plots():
    global folder
    filename = folder / "test_dont_display_output.pdf"
    pdf = PDFSaver(filename, 3, 4, "Dont show the images in the API", False)
    for i in range(8):
        ax = pdf.get_ax()
        _plot_random(ax)
    pdf.close_pdf()


if __name__ == "__main__":
    print(folder.cwd())
    folder.mkdir(exist_ok=True)
    test_one_ax()
    test_one_complete_page()
    test_general_case()
    test_option_dont_display_plots()
