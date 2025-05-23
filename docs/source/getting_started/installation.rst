Installation
############

.. warning::
    This project is under active development.

Installing ``mejiro`` and its dependencies
******************************************

Create a conda environment:

.. code-block:: bash    

    conda env create -f environment.yml

.. Install ``SLSim``:

.. .. code-block:: bash  

..     git clone https://github.com/LSST-strong-lensing/slsim.git
..     cd slsim
..     pip install -e .

To generate PSFs with ``STPSF`` (formerly ``WebbPSF``), follow the instructions `here <https://stpsf.readthedocs.io/en/latest/installation.html>`__ to download the required data files and set environment variables.

Optional setup: Pandeia
========================

Install Pandeia by following the
instructions `here <https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation>`__.

First-time setup
****************

1. Duplicate any yaml file in the ``config/machine`` directory and rename it, then update ``repo_dir`` and ``data_dir``
   attributes to point to the directories where the local repository resides and output data should be written,
   respectively.
2. Copy ``setup/config.yaml`` to ``config/config.yaml``
3. Modify the ``defaults.machine`` attribute in the file you just copied (``config/config.yaml``) to the name (without extension) of the yaml file created in step 1.

.. warning::
   Note that this ``hydra``-based configuration will be changed soon.