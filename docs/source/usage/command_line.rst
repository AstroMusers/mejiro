Running Steps from the Command Line
===================================

The ``Pipeline`` class (see :doc:`running`) runs every step in order. You can also run the steps one at a time, which is useful for re-running a single step, running on a cluster, or running instruments that ``Pipeline.run()`` doesn't support end to end (see the instrument list under ``instrument`` in :doc:`configuration`).

Each step is a script in ``mejiro/pipeline/``. From the root of the repository:

.. code-block:: bash

    python mejiro/pipeline/_00_cache_psfs.py --config path/to/config.yaml --data_dir path/to/output

``examples/execute_pipeline.sh`` runs every step this way.

Flags shared by every step
--------------------------

``--config``
   Path to the YAML configuration file. Required.

``--data_dir``
   Parent directory for pipeline output. Overrides ``data_dir`` in the config, and is required when the config leaves it ``null`` (as ``simple.yaml`` and ``test.yaml`` do).

``--resume``
   Keep the step's existing output and skip items that are already done. **Without it, a step deletes its existing output and starts from scratch.** Not available for ``_00_cache_psfs.py`` (which always skips PSFs that are already cached) or ``_06_h5_export.py``.

``--sequential``
   Process systems one at a time, in order, instead of in parallel. When ``limit`` is set, take the first systems rather than a random sample. Useful for debugging. Available for steps 03, 04, 05 and ``calculate_snrs.py``.

The steps
---------

``_00_cache_psfs.py``
   Compute and cache Roman PSFs with STPSF. Roman only.

``_01a_generate_galaxy_tables.py``
   Draw galaxy population tables with SkyPy.

``_01b_run_survey_simulation.py``
   Find detectable strong lenses with SLSim.

``_02_build_lens_list.py``
   Convert them into mejiro ``StrongLens`` objects.

``_03_generate_subhalos.py``
   Add dark-matter substructure with pyHalo. Roman and HWO only.

``_04_create_synthetic_images.py``
   Ray-shoot the synthetic images. Also takes ``--prev-step {02,03}`` (default ``03``): the step to read lenses from. Use ``02`` to skip substructure, or for JWST, which has no step 03.

``_05_galsim.py``
   Simulate exposures with GalSim.

``_05_romanisim.py``
   Simulate Roman exposures with romanisim, instead of ``_05_galsim.py``. Roman only. Also takes:

   - ``--level {l2,l3}`` (default ``l2``): single-exposure (L2) cutouts, or cutouts from co-added L3 mosaics.
   - ``--dither-pattern``: the dither pattern for ``--level l3`` (default ``BOXGAP4_1``). Sub-pixel patterns such as ``SUB4`` are also accepted; see ``mejiro/data/roman_dither_patterns/``.
   - ``--max-systems N``: process only the first ``N`` systems (lowest IDs) across all detectors.

``calculate_snrs.py``
   Calculate each system's signal-to-noise ratio. Roman and HWO only.

``_06_h5_export.py``
   Export the dataset to an HDF5 file (see :doc:`outputs`). Roman only.

``calculate_snrs.py`` and ``_06_h5_export.py`` also take ``--prev-step``: the step whose exposures to read. **It defaults to** ``05_romanisim``, **so after** ``_05_galsim.py``, **pass** ``--prev-step 05_galsim``. (``Pipeline`` sets this for you.)

The order is 00, 01a, 01b, 02, 03, 04, then one of:

- ``_05_galsim.py``, then ``_06_h5_export.py --prev-step 05_galsim``
- ``_05_romanisim.py``, then ``calculate_snrs.py``, then ``_06_h5_export.py``

Running one step from Python
----------------------------

``Pipeline.run_script()`` runs a single step with the pipeline's config, e.g. ``pipeline.run_script(4)`` or ``pipeline.run_script('1b')``. See :class:`mejiro.pipeline.mejiro_pipeline.Pipeline` for the accepted values.

Maintenance tools
-----------------

``scan_corrupted_pickles.py <directory>``
   Find pickles that can't be loaded (e.g. from an interrupted write) under a directory. ``--pattern`` sets the filename pattern (default ``SyntheticImage_*.pkl``), ``--delete`` deletes them, and ``--workers`` sets the number of processes. Re-running the step with ``--resume`` then regenerates only the missing files.

``compact_03_realizations.py --config ...``
   Strip the pyHalo realization objects from existing step 03 output, as ``subhalos.serialization: lightweight`` would have. Takes ``--data_dir``, ``--workers`` and ``--dry-run``.

``make_substructure_csv.py --config ...``
   Write ``substructure_flags.csv`` (see :doc:`outputs`) from existing step 03 output: one row per system with its ``id`` and ``substructure_flag``. Takes ``--data_dir``.
