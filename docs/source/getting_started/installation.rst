Installation
============

Install ``mejiro`` from `GitHub <https://github.com/AstroMusers/mejiro>`__. Currently, ``mejiro`` is not available on PyPI.

.. code-block:: bash

    git clone https://github.com/AstroMusers/mejiro.git
    cd mejiro
    conda create -n mejiro python
    conda activate mejiro
    pip install -e .

The following optional dependencies can be installed for additional features:

- ``dev``: run unit tests and build documentation
- ``jax``: ``jaxtronomy`` and ``jax[cuda12]`` for faster survey simulation and ray-shooting steps
- ``roman``: simulate Roman PSFs with ``STPSF`` and Roman L2 and L3 data products with ``romanisim`` and ``romancal``
- ``hwo``: simulate HWO exposures with ``syotools``
- ``all``: install all optional dependencies

.. code-block:: bash

    pip install -e .[all]

Note: if you are using zsh, the default shell on macOS, you will need to escape the square brackets, e.g., ``pip install -e .'[all]'``.

.. Installing ``jaxtronomy`` can provide significant speed-ups for the survey simulation with `SL-Hammocks` and ray-shooting steps, especially if your machine has GPUs. To install ``jaxtronomy``, follow the instructions `here <https://github.com/lenstronomy/JAXtronomy>`__.

To simulate images from *Roman*, you will need to download the ``roman-technical-information`` repository `here <https://github.com/RomanSpaceTelescope/roman-technical-information/>`__. Then, set the environment variable ``ROMAN_TECHNICAL_INFORMATION_PATH`` to the path where you downloaded the repository, e.g., in your ``.bashrc`` or ``.bash_profile``:

.. code-block:: bash

    export ROMAN_TECHNICAL_INFORMATION_PATH="/{your_path}"

To simulate images from *HWO*, `follow these instructions <https://github.com/spacetelescope/syotools>`__ for ``syotools`` to set the environment variables ``PYSYN_CDBS`` and ``SCI_ENG_DIR``.

To generate PSFs with ``STPSF`` (formerly ``WebbPSF``), follow the instructions `here <https://stpsf.readthedocs.io/en/latest/installation.html>`__ to download the required data files and set environment variables.

.. Optional setup: Pandeia
.. ========================

.. Install Pandeia by following the
.. instructions `here <https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation>`__.
