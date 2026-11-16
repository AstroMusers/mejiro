Running on the Roman Research Nexus
===================================

The `Roman Research Nexus <https://roman-docs.stsci.edu/data-handbook/roman-research-nexus>`__ (RRN) is STScI's JupyterHub for Roman. It comes with the Roman software, the CRDS calibration reference files and the STPSF data that ``mejiro`` uses, and it can stream the SOC's simulated Roman surveys from AWS S3. This page covers what is different about installing and running ``mejiro`` there. Everything else in these docs applies unchanged.

What the Nexus provides
-----------------------

- **Kernels** with STScI's Roman software (``romanisim``, ``romancal``, ``stpsf``, ``crds``, ...) preinstalled.
- **Reference data and environment variables.** ``CRDS_SERVER_URL`` and ``CRDS_PATH`` are set, and so is ``STPSF_PATH``, pointing at preinstalled STPSF data. ``mejiro`` reads none of these itself: STPSF, romanisim, romancal and CRDS do, so whatever the Nexus sets is what ``mejiro`` uses.
- **Storage.** Your home directory and team directories persist between sessions. The shared tutorial notebooks are read-only and are re-synced when you log in, so keep your own work outside them.
- **Servers** of several sizes, chosen when the server starts: small (2 vCPU, 16 GB), medium (8 vCPU, 16 GB), and large (32 vCPU, 64 GB, or 16 vCPU, 128 GB; team accounts only). Servers stop after about an hour without activity.

Install ``mejiro`` in its own kernel
------------------------------------

Don't ``pip install`` into the default Nexus kernel. Installs there don't persist (they are gone the next time you log in), and ``mejiro`` pins exact versions of ``romanisim``, ``romancal``, ``stpsf`` and ``astropy``, which would replace the versions the default kernel was built with. Instead, make a kernel for ``mejiro`` with the Nexus's ``kernel-create`` helper (see `Installing Extra Software <https://spacetelescope.github.io/roman_notebooks/markdown/software.html>`__). In a terminal (**File → New → Terminal**):

.. code-block:: bash

    kernel-create mejiro 3.13 "mejiro"
    source kernel-activate mejiro

    cd ~
    git clone https://github.com/AstroMusers/mejiro.git
    cd mejiro
    pip install -e ".[dev,roman]"

Then choose the **mejiro** kernel for your notebooks. Run ``source kernel-activate mejiro`` in each new terminal before running ``mejiro`` from the command line. ``mejiro`` supports Python 3.12 and 3.13 (see :doc:`installation`). Installing needs access to GitHub, because ``slsim``, ``speclite`` and ``skypy`` are installed from GitHub forks.

Use the Nexus's reference data
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Data
     - Environment variables
     - Used by
   * - CRDS reference files
     - ``CRDS_PATH``, ``CRDS_SERVER_URL``, and optionally ``CRDS_CONTEXT``
     - ``_05_romanisim.py --level l3`` (romancal's ``MosaicPipeline``), and ``_05_romanisim.py`` at either level when ``exposure.usecrds`` is ``true`` (see :doc:`configuration`).
   * - STPSF data
     - ``STPSF_PATH``
     - ``_00_cache_psfs.py``, and any other step that has to compute a PSF that isn't cached yet.
   * - Roman instrument parameters
     - none
     - Everything that simulates Roman images. They come from the ``roman-technical-information`` package, which ``pip`` installs into the ``mejiro`` kernel.

Check what is set, from a terminal with the ``mejiro`` kernel activated:

.. code-block:: bash

    env | grep -E 'CRDS|STPSF'
    crds list --status           # CRDS server, cache, and the context in use
    cat $STPSF_PATH/version.txt  # the STPSF data version
    python -c "import stpsf; print(stpsf.DATA_VERSION_MIN)"

**STPSF data version.** The pinned ``stpsf`` refuses data older than its ``DATA_VERSION_MIN``. If the Nexus's preinstalled data is older, download the data that matches (see the `STPSF installation instructions <https://stpsf.readthedocs.io/en/latest/installation.html>`__), unpack it under your home directory, e.g. ``~/refdata/stpsf-data``, and point ``STPSF_PATH`` there for the ``mejiro`` kernel:

- For terminals, add ``export STPSF_PATH=~/refdata/stpsf-data`` to ``~/.bashrc``.
- For notebooks, either add ``"env": {"STPSF_PATH": "/home/<you>/refdata/stpsf-data"}`` to the kernel's ``kernel.json`` (``jupyter kernelspec list`` shows where it is), or set ``os.environ['STPSF_PATH']`` at the top of the notebook, before anything imports ``stpsf``.

**CRDS cache.** romancal and romanisim download reference files into ``CRDS_PATH`` the first time they need them, and on the Nexus that cache is in your home directory. With ``exposure.usecrds: true``, step 05 downloads the dark, linearity, flat and other reference files for every detector and band it simulates before it starts, which can come to several GB per detector. Building L3 coadds on skycells, as the HLWAS notebook below does, also downloads romancal's skymap reference file once (about 1.3 GB). ``du -sh $CRDS_PATH`` shows how much is there. ``CRDS_CONTEXT``, if set, pins the context; otherwise CRDS uses the server's current one. ``_05_romanisim.py`` records the context it used in ``05_romanisim/crds_context.txt``, and the HDF5 dataset records it as ``crds_context`` (see :doc:`outputs`).

Run the unit tests
------------------

From the repository, with the ``mejiro`` kernel activated, run the suite the way CI does (the HWO tests need the ``hwo`` optional dependency):

.. code-block:: bash

    cd ~/mejiro
    pytest -v --ignore tests/test_hwo_imaging --ignore tests/test_engines/test_galsim_engine_hwo.py --ignore tests/test_instruments/test_hwo.py

None of these tests need CRDS, the STPSF data or a network connection, and the test configurations use one or two cores, so any server size will do. To also check that ``mejiro`` can use the Nexus's CRDS setup, run the tests marked ``crds``. They download one detector's reference files into ``CRDS_PATH`` and simulate one L2 exposure with them:

.. code-block:: bash

    pytest --run-crds tests/test_pipeline/test_romanisim_crds.py

Run the pipeline
----------------

:doc:`running` and :doc:`command_line` apply as they are. On the Nexus:

- Put ``data_dir`` in your home directory or a team directory, so the output persists.
- Keep each ``cores.*`` value in the config at or below the server's vCPU count, and the per-worker memory within the server's RAM. ``simple.yaml``, for example, asks for 36 cores for step 00, more than any Nexus server has.
- Set ``exposure.usecrds: true`` to simulate step 05's exposures with the CRDS reference files rather than romanisim's built-in defaults.
- If the server stops partway through a long step, start it again and rerun the step with ``--resume`` (see :doc:`command_line`).

Inject lenses into the SOC's HLWAS simulations
----------------------------------------------

The SOC's simulations of the Core Community Surveys, described in STScI's `CCS simulations tutorial <https://spacetelescope.github.io/roman_notebooks/notebooks/ccs_simulations/ccs_simulations.html>`__, are on S3 at ``s3://stpubdata/roman/nexus/soc_simulations/``. They include about 6 square degrees of the High-Latitude Wide-Area Survey (HLWAS) as L2 exposures and L3 coadds, along with the input catalogs they were simulated from. Everything needed to stream them (``s3fs``, ``astropy_healpix``, ``crds``) comes with the ``roman`` optional dependency.

The ``lens_search_hlwas.ipynb`` notebook in the ``mejiro-projects`` repository puts ``mejiro`` strong lenses into these images:

#. It matches galaxies in the HLWAS input catalog to deflectors from a ``mejiro`` run's step 02, on magnitude and colors.
#. It renders only the lensed source of each match with ``mejiro``, rotated to the galaxy's position angle, so the galaxy the SOC already simulated is the deflector.
#. It injects the arcs into every L2 exposure that covers one skycell with romanisim, using each exposure's own PSF, gain and read noise from CRDS.
#. It rebuilds that skycell's L3 coadd with romancal, both from the original and from the injected L2s, so their difference shows the injected arcs alone.

It needs a medium or large server.
