Installation
############

Installing ``mejiro`` and its dependencies
******************************************

Download ``mejiro`` from `GitHub <https://github.com/AstroMusers/mejiro>`__.

Create a conda environment:

.. code-block:: bash    

    conda env create -f environment.yml

To simulate images from *Roman*, you will need to download the ``roman-technical-information`` repository `here <https://github.com/spacetelescope/roman-technical-information>`__. Then, set the environment variable ``ROMAN_TECHNICAL_INFORMATION_PATH`` to the path where you downloaded the repository, e.g., in your ``.bashrc``:

.. code-block:: bash

    export ROMAN_TECHNICAL_INFORMATION_PATH="/{your_path}/roman-technical-information"

To generate PSFs with ``STPSF`` (formerly ``WebbPSF``), follow the instructions `here <https://stpsf.readthedocs.io/en/latest/installation.html>`__ to download the required data files and set environment variables.

Optional setup: ``hwo-tools``
=============================

Install `hwo-tools` by following the
instructions `here <https://github.com/spacetelescope/hwo-tools/blob/main/README.md>`__.

.. Optional setup: Pandeia
.. ========================

.. Install Pandeia by following the
.. instructions `here <https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation>`__.
