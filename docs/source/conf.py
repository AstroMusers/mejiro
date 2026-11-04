# Configuration file for the Sphinx documentation builder.

import datetime
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

# -- Project information

project = 'mejiro'
copyright = f'2023-{datetime.date.today().year}, AstroMusers'  # first commit: 2023-10-30
author = 'Bryce Wedig'

import mejiro

try:
    # Read the Docs builds from the source tree without installing mejiro, so there's no
    # mejiro/_version.py there; ask git for the version directly.
    from setuptools_scm import get_version
    release = get_version(root=REPO_ROOT)
except Exception:
    release = mejiro.__version__
version = release

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',  # generate autodocs
    'sphinx.ext.napoleon',  # for autodoc config
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',  # autodocs with math
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode'  # link to source code
]

autosectionlabel_prefix_document = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True  # for math

templates_path = ['_templates']

# https://stackoverflow.com/questions/67485175/docstrings-are-not-included-in-read-the-docs-sphinx-build/67486947#67486947
autodoc_mock_imports = ['numpy', 'pandas', 'PIL', 'astropy', 'galsim', 'stpsf', 'tqdm', 'lenstronomy',
                        'slsim', 'speclite', 'specutils', 'synphot', 'pyHalo', 'pysiaf', 'pandeia',
                        'yaml', 'scipy', 'syotools', 'h5py', 'matplotlib', 'romanisim',
                        'asdf', 'roman_datamodels', 'romancal', 'stcal']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Module dependency diagram (appendices/structure.rst)

def generate_structure_diagram(app):
    """Draw the module dependency diagram with pydeps, which needs graphviz's ``dot``.

    The diagram is generated on every build rather than committed, so it can't go out of
    date. Pipeline scripts are left out to keep it readable.
    """
    if shutil.which('dot') is None:
        # SystemExit passes through Sphinx's error handling, so the build stops with just this
        # message instead of a crash report asking for a Sphinx bug report.
        raise SystemExit(
            "Error: building the docs needs graphviz's 'dot' program to draw the module "
            "diagram in appendices/structure.rst. Install graphviz (e.g. 'apt install graphviz', "
            "'brew install graphviz' or 'conda install graphviz') and build again."
        )
    out_dir = os.path.join(app.srcdir, '_generated')
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [sys.executable, '-m', 'pydeps', os.path.join(REPO_ROOT, 'mejiro'),
         '--only', 'mejiro', '--exclude', 'mejiro.pipeline*', '--noshow',
         '-o', os.path.join(out_dir, 'mejiro.svg')],
        check=True, cwd=out_dir,
    )


def setup(app):
    app.connect('builder-inited', generate_structure_diagram)
