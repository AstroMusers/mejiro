# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import mejiro

# -- Project information

project = 'mejiro'
copyright = '2024, AstroMusers'
author = 'Bryce Wedig'

release = '0.0'
version = '0.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',  # generate autodocs
    'sphinx.ext.napoleon',  # for autodoc config
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',  # autodocs with math
    'sphinx.ext.autosectionlabel'
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
autodoc_mock_imports = ['numpy', 'pandas', 'PIL', 'astropy', 'omegaconf', 'galsim', 'webbpsf', 'tqdm', 'lenstronomy', 'slsim', 'speclite', 'pyhalo', 'pysiaf', 'pandeia-engine', 'hydra-core', 'yaml', 'scipy']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'