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
- ``jax-cpu``: ``jaxtronomy`` plus CPU-only ``jax`` for faster survey simulation and ray-shooting on machines without a GPU
- ``jax-gpu``: ``jaxtronomy`` plus ``jax[cuda12]`` for GPU-accelerated survey simulation and ray-shooting. Requires a CUDA-12-compatible NVIDIA driver on the host (``nvidia-smi`` should run); ``jax-gpu`` will install on a CPU-only machine but JAX will fall back to CPU at runtime with a warning.
- ``roman``: simulate Roman PSFs with ``STPSF`` and Roman L2 and L3 data products with ``romanisim`` and ``romancal``
- ``hwo``: simulate HWO exposures with ``syotools``
- ``all``: install all optional dependencies (assumes a GPU host; uses ``jax-gpu``)

.. code-block:: bash

    pip install -e .[all]

Note: if you are using zsh, the default shell on macOS, you will need to escape the square brackets, e.g., ``pip install -e .'[all]'``.

Selecting the JAX backend at runtime: set ``JAX_PLATFORM_NAME=cpu`` or ``JAX_PLATFORM_NAME=gpu`` in the shell before launching Python (the variable must be set *before* ``jax`` is imported). The pipeline scripts read ``jaxtronomy.jax_platform`` from the YAML config and set this variable for you.

To simulate images from *Roman*, ``mejiro`` reads instrument parameters from the ``roman-technical-information`` package, which is installed automatically as part of the ``roman`` optional dependency (``pip install -e .[roman]``). No manual download or environment variable is required.

To simulate images from *HWO*, `follow these instructions <https://github.com/spacetelescope/syotools>`__ for ``syotools`` to set the environment variables ``PYSYN_CDBS`` and ``SCI_ENG_DIR``.

To generate PSFs with ``STPSF`` (formerly ``WebbPSF``), follow the instructions `here <https://stpsf.readthedocs.io/en/latest/installation.html>`__ to download the required data files and set environment variables.

.. Optional setup: Pandeia
.. ========================

.. Install Pandeia by following the
.. instructions `here <https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation>`__.
