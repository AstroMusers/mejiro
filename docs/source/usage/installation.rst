Installation
============

Install ``mejiro`` from `GitHub <https://github.com/AstroMusers/mejiro>`__. Currently, ``mejiro`` is not available on PyPI. ``mejiro`` supports Python 3.12 and 3.13 (CI tests 3.13).

.. code-block:: bash

    git clone https://github.com/AstroMusers/mejiro.git
    cd mejiro
    conda create -n mejiro python=3.13
    conda activate mejiro
    pip install -e ".[roman]"

Most uses need the ``roman`` optional dependency, as the table below shows. On its own, ``pip install -e .`` installs the core packages but can't simulate images for any instrument. Keep the quotes around ``".[roman]"``: they stop zsh (the default shell on macOS) from treating the square brackets as a pattern, and they work in bash too.

Which optional dependencies do I need?
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - To...
     - Install
     - Why
   * - Run the pipeline with the ``Pipeline`` class, for any instrument
     - ``roman``
     - ``Pipeline`` imports every pipeline step, and ``_00_cache_psfs``, ``_05_romanisim`` and ``_06_h5_export`` import ``stpsf``, ``romanisim`` and ``romancal``.
   * - Simulate *Roman* images
     - ``roman``
     - Instrument parameters come from ``roman-technical-information`` and PSFs from ``STPSF``.
   * - Simulate *JWST* images
     - ``roman``
     - JWST PSFs also come from ``STPSF``, which is part of this optional dependency.
   * - Simulate *HWO* images
     - ``hwo``, plus ``roman`` to use the ``Pipeline`` class
     - ``syotools`` provides the HWO telescope and camera models.
   * - Run the unit tests or build the documentation
     - ``dev``
     -
   * - Use an NVIDIA GPU for JAX
     - ``jax-gpu``
     - See the ``jax-gpu`` entry below.

The optional dependencies:

- ``dev``: run unit tests and build documentation (``pytest``, ``sphinx``, ``jupyter``).
- ``jax-gpu``: ``jax[cuda12]`` for GPU-accelerated survey simulation and ray-shooting (``jax`` and ``jaxtronomy`` are installed by default and run on CPU). Requires a CUDA-12-compatible NVIDIA driver on the host (``nvidia-smi`` should run); ``jax-gpu`` will install on a CPU-only machine but JAX will fall back to CPU at runtime with a warning.
- ``roman``: ``STPSF`` for Roman and JWST PSFs, ``romanisim`` and ``romancal`` for Roman L2 and L3 data products, ``roman-technical-information`` for Roman instrument parameters, ``pysiaf`` for Roman WFI pointing geometry, and ``astrocut`` for image cutouts.
- ``hwo``: ``syotools`` for HWO telescope and camera models.
- ``all``: install all optional dependencies (assumes a GPU host; uses ``jax-gpu``).

To install several at once, list them together, e.g.:

.. code-block:: bash

    pip install -e ".[roman,hwo]"
    pip install -e ".[all]"

Selecting the JAX backend at runtime: set ``JAX_PLATFORM_NAME=cpu`` or ``JAX_PLATFORM_NAME=gpu`` in the shell before launching Python (the variable must be set *before* ``jax`` is imported). The pipeline scripts read ``jaxtronomy.jax_platform`` from the YAML config and set this variable for you.

To simulate images from *Roman*, ``mejiro`` reads instrument parameters from the ``roman-technical-information`` package, which is installed automatically as part of the ``roman`` optional dependency (``pip install -e ".[roman]"``). No manual download or environment variable is required.

To simulate images from *HWO*, `follow these instructions <https://github.com/spacetelescope/syotools>`__ for ``syotools`` to set the environment variables ``PYSYN_CDBS`` and ``SCI_ENG_DIR``.

To generate PSFs with ``STPSF`` (formerly ``WebbPSF``), follow the instructions `here <https://stpsf.readthedocs.io/en/latest/installation.html>`__ to download the required data files and set environment variables.

.. Optional setup: Pandeia
.. ========================

.. Install Pandeia by following the
.. instructions `here <https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation>`__.
