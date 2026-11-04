Pipeline Outputs
================

Directory layout
----------------

All output goes under ``<data_dir>/<pipeline_label>/`` (``<pipeline_label>_dev/`` when ``dev: True``), with one subdirectory per step, named after the step:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Directory
     - Contents
   * - ``01a/``
     - Galaxy population tables: ``galaxy_table_<NNNN>.pkl``.
   * - ``01b/``
     - Detectable lenses per run: ``detectable_gglenses_<run>.pkl``, plus ``run_complete_<run>.txt`` markers used by ``--resume``. With ``survey.write_to_csv``, also ``detectable_pop_<run>.csv``, and ``total_pop_<run>.csv`` when ``survey.total_population`` is set.
   * - ``02/``
     - One pickled ``StrongLens`` per system: ``lens_<name>.pkl``.
   * - ``03/``
     - The same lenses with substructure added: ``lens_<name>.pkl``. Systems pyHalo couldn't handle are saved as ``failed_<name>.pkl``.
   * - ``04/``
     - Synthetic images: ``SyntheticImage_<name>_<band>.pkl``, or ``.npz`` with ``synthetic_image.serialization: lightweight``. Failures are saved as ``failed_<name>_<band>.pkl``.
   * - ``05_galsim/``
     - Exposures from ``_05_galsim.py``: ``Exposure_<name>_<band>.pkl``, or ``.npz`` with ``imaging.serialization: lightweight``.
   * - ``05_romanisim/``
     - Exposure cutouts from ``_05_romanisim.py``: ``Exposure_<name>_<band>.npy``. Also ``exposure_level.txt`` (``l2`` or ``l3``), ``exposure_time.txt`` (``--level l3`` only: the co-added depth in seconds), and per-batch progress files. L3 runs keep intermediate mosaics under ``<data_dir>/_scratch/``.
   * - ``snr/``
     - ``name_snr_pairs.pkl``: each system's SNR, from ``calculate_snrs.py``.
   * - ``06/``
     - The dataset, ``<pipeline_label>_v_<version>.h5`` (``dataset.version``, with ``.`` replaced by ``_``). When ``dataset.labeled`` is ``False``, also ``<pipeline_label>_answer_key_v_<version>.csv``.

For Roman, the files from steps 02 to 05 are split into one subdirectory per detector (``sca01/`` to ``sca18/``); for other instruments they sit directly in the step's directory.

Each system is named ``<pipeline_label>_<uid>``, where ``<uid>`` is an eight-digit ID assigned in step 02, e.g. ``simple_00000042``.

PSFs cached by step 00 go to ``psf_cache_dir``, not under ``data_dir``, named ``<band>_<detector>_<x>_<y>_<oversample>_<num_pix>.npy`` (for JWST, ``<band>_<oversample>_<num_pix>.npy``).

To load step 04 and 05 files in Python, use ``mejiro.utils.util.load_synthetic_image`` and ``mejiro.utils.util.load_exposure``, which handle both the ``.pkl`` and ``.npz`` formats.

The HDF5 dataset
----------------

``_06_h5_export.py`` writes one HDF5 file. Most attributes are stored as a two-element array, ``(value, description)``.

**File attributes:** ``author`` (``user@host``), ``created``, ``dataset_version``, ``mejiro_version``, ``lenstronomy_version``, ``slsim_version``, ``romanisim_version``, ``stpsf_version``, and ``romancal`` (the romancal version).

**Systems:** the ``images`` group holds one group per system, ``strong_lens_<uid>``, with these attributes:

- Always: ``uid``, ``z_source``, ``z_lens``, ``mu`` (flux-weighted magnification of the source), ``detector``, ``detector_position_x``, ``detector_position_y``.
- When ``dataset.labeled`` is ``True``: ``main_halo_mass`` (M\ :sub:`sun`), ``theta_e`` (Einstein radius, arcsec), ``sigma_v`` (lens velocity dispersion, km/s), ``substructure`` (``True``/``False``), the ``subhalos.realization_kwargs`` values for systems with substructure, and ``deflector_only``.

**Images:** each system group holds one dataset per band in ``synthetic_image.bands``:

- ``exposure_<uid>_<band>``: the simulated exposure. Attributes: ``exposure_time`` (seconds), ``pixel_scale`` (arcsec/pixel), ``fov`` (arcsec), ``units``, ``filter``, ``source_magnitude`` (unlensed), ``lensed_source_magnitude``, ``lens_magnitude``, and ``snr`` if ``calculate_snrs.py`` ran.
- ``synthetic_image_<uid>_<band>``, when ``dataset.include_synthetic_images`` is ``True``: the noise-free synthetic image, with ``pixel_scale``, ``fov``, ``filter``, the three magnitudes, and ``units``, which is always ``counts/s``: counts per second in each pixel of the dataset's own grid. That grid is finer than the exposure's when ``synthetic_image.oversample`` is greater than 1; see its ``pixel_scale``.

The exposures' units and exposure time depend on the step that made them:

- ``_05_galsim.py``: ``DN``, with ``imaging.exposure_time``.
- ``_05_romanisim.py --level l2``: ``DN/s``, with the exposure time of the MA table in ``exposure.ma_table_number``.
- ``_05_romanisim.py --level l3``: ``MJy/sr``, with the co-added depth.

.. note::

   Files exported by ``mejiro`` v3.0.0 and later, before this was fixed, give ``synthetic_image_*`` datasets the exposure's ``units`` (``Counts``, ``DN``, ``DN/s`` or ``MJy/sr``). Their pixel values are counts/s all the same; only the label is wrong.

**Answer key:** when ``dataset.labeled`` is ``False``, the truth values are left out of the HDF5 file, and the answer-key CSV has one row per system with the columns ``uid``, ``einstein_radius``, ``substructure_flag`` and ``deflector_only``.
