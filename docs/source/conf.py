import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "mdmpy"
copyright = "2023, mdmpy Authors"
author = "mdmpy Authors"

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_dark_mode",
]

bibtex_bibfiles = ["references.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}
