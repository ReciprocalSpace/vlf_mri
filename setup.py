"""Setup file for the VLF-MRI library."""

import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="vlf_mri",
    version="0.0.1",
    description="Library for VLF-MRI project",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/ReciprocalSpace/vlf_mri",
    author="Aimé Labbé and Marie Poirier-Quinot",
    author_email="aime.labbe@universite-paris-saclay.fr",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    packages=setuptools.find_packages(
        exclude=[
            "code"
        ],
        include=["vlf_mri"]

    ),
    # entry_points={"console_scripts": ["tree-cli=trees.bin.tree_cli:main"]},
    python_requires=">=3.7",
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'tqdm>=4.62',
        'matplotlib>=3.4',
        'lmfit>=1.0'
    ]
)
