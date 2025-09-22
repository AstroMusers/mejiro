# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'mejiro'
copyright = '2025, AstroMusers'
author = 'Bryce Wedig'

import mejiro

release = mejiro.__version__
version = mejiro.__version__

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',  # generate autodocs
    'sphinx.ext.napoleon',  # for autodoc config
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',  # autodocs with math
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode'  # link to source code
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True  # for math

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# https://stackoverflow.com/questions/67485175/docstrings-are-not-included-in-read-the-docs-sphinx-build/67486947#67486947
autodoc_mock_imports = ['numpy', 'pandas', 'PIL', 'astropy', 'galsim', 'stpsf', 'tqdm', 'lenstronomy',
                        'slsim', 'speclite', 'specutils', 'synphot', 'pyhalo', 'pysiaf', 'pandeia-engine',
                        'yaml', 'scipy', 'syotools', 'h5py']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
