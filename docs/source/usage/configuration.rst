Configuration File Reference
============================

The mejiro pipeline is driven entirely by a YAML configuration file. Example configurations live in ``mejiro/data/mejiro_config/`` (e.g., ``simple.yaml``, ``roman_test.yaml``, ``roman_data_challenge_rung_1.yaml``); the recommended starting point is to copy one and edit it.

This page documents every top-level section and attribute. Units are given only where they are documented in the YAML comments or in the code. Pipeline scripts referenced below all live in ``mejiro/pipeline/``.

Global attributes
-----------------

These keys sit at the top level of the YAML file.

``data_dir``
   Base directory under which all pipeline output is written. Required: if ``null``, the ``--data_dir`` CLI argument must be supplied to override it. The per-pipeline output directory is ``<data_dir>/<pipeline_label>``.

   Used by ``PipelineHelper`` to set ``pipeline_dir`` for every pipeline script, and read directly in ``romanisim_pipeline.py`` to locate input synthetic images and write romanisim output.

``pipeline_label``
   Name of the pipeline run. Used to construct the output directory (``<data_dir>/<pipeline_label>/``) and as a prefix for output filenames. When ``dev: True``, ``_dev`` is appended.

   Used in every pipeline script via ``PipelineHelper``. In ``_01a_generate_galaxy_tables.py`` and ``_01b_run_survey_simulation.py`` the literal value ``all`` triggers loading of every supported instrument's speclite filters.

``psf_cache_dir``
   Directory where cached STPSF PSFs are stored as ``.npy`` files. May be a path relative to ``data_dir``, an absolute path, or ``null``. When ``null``, ``PipelineHelper`` defaults to ``mejiro/data/psfs/<instrument>``.

   Used in ``_00_cache_psfs.py`` as the write target, and in ``_01b_run_survey_simulation.py``, ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, ``_04_jax_create_synthetic_images.py``, ``calculate_snrs.py``, and ``_06_h5_export.py`` as the read target when constructing ``kwargs_psf``.

``dev``
   Development-mode flag. When ``True``, ``_dev`` is appended to ``pipeline_label`` so dev runs do not overwrite production output.

   Used by ``PipelineHelper`` and explicitly by ``romanisim_pipeline.py`` to set the pipeline directory.

``nice``
   Process ``nice`` value applied to every pipeline worker. Defaults to ``0`` if absent.

   Set via ``os.nice(...)`` in ``PipelineHelper.__init__``.

``show_progress_bar``
   Boolean. When ``True``, tqdm bars are shown during long inner loops (e.g., the per-run candidate loops in ``_01b_run_survey_simulation.py``).

   Used in ``_01b_run_survey_simulation.py`` to toggle the per-candidate progress bars.

``suppress_warnings``
   Boolean. When ``True``, ``warnings.filterwarnings("ignore")`` is installed in each worker process.

   Used in ``_01a_generate_galaxy_tables.py`` and ``_01b_run_survey_simulation.py`` inside the worker functions; also set globally in ``PipelineHelper`` to suppress ``UserWarning``.

``logging_level``
   String passed to ``logging.basicConfig`` (e.g., ``INFO``, ``WARNING``, ``DEBUG``). Defaults to ``INFO`` if absent.

   Set in ``PipelineHelper.__init__``.

``limit``
   Maximum number of systems each script should process, or ``null`` for no limit.

   Used in ``_01b_run_survey_simulation.py`` to cap the number of detectable lenses per run, in ``_02_build_lens_list.py`` to short-circuit the lens-conversion loop, in ``_03_generate_subhalos.py``, ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, ``_04_jax_create_synthetic_images.py``, ``_05_galsim.py``, and ``calculate_snrs.py`` to subsample input pickles (sequentially with ``--sequential``, otherwise via ``np.random.choice``), and in ``romanisim_pipeline.py`` to subsample lens IDs per SCA.

``seed``
   Integer global random seed. Defaults to ``42`` when accessed via ``config.get('seed', 42)``.

   Used in ``_01a_generate_galaxy_tables.py`` to seed the per-table draw via ``hash((seed, 'table', table_index))``, in ``_01b_run_survey_simulation.py`` for the per-run draw via ``hash((seed, run))``, in ``_03_generate_subhalos.py`` to seed the substructure assignment mask, and in ``romanisim_pipeline.py`` to derive per-batch RNGs (``batch_seed = seed + sca_num * 10000 + band_idx * 1000 + batch_idx``). Typically also referenced from ``imaging.engine_params.rng_seed`` via a YAML anchor.

``cores``
-----------

Per-script worker counts for ``ProcessPoolExecutor``. Each key maps the script name to its worker count; ``PipelineHelper.calculate_process_count`` reads ``cores.script_<script_name>``.

``script_00``
   Workers for ``_00_cache_psfs.py``.

``script_01a``
   Workers for ``_01a_generate_galaxy_tables.py``.

``script_01b``
   Workers for ``_01b_run_survey_simulation.py``.

``script_02``
   Workers for ``_02_build_lens_list.py`` (single-process in current scripts; included for completeness).

``script_03``
   Workers for ``_03_generate_subhalos.py``.

``script_04``
   Workers for ``_04_create_synthetic_images.py`` (and its interpol/JAX variants).

``script_05``
   Workers for ``_05_galsim.py``.

``script_05_romanisim``
   Workers for ``romanisim_pipeline.py``. Read directly as ``config['cores']['script_05_romanisim']``; the per-worker thread count is derived as ``max(2, 64 // num_workers)``.

``script_snr``
   Workers for ``calculate_snrs.py``.

``jaxtronomy``
--------------

Controls JAX-based acceleration for ray-shooting and image rendering.

``use_jax``
   Boolean. When ``True``, lens objects are built with ``use_jax=True`` so that ``jaxtronomy`` is used in place of ``lenstronomy`` for ray-shooting.

   Used in ``_01b_run_survey_simulation.py``, ``_02_build_lens_list.py``, ``_03_generate_subhalos.py``, ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py``. In ``_04_create_synthetic_images.py`` it also gates whether ``JAX_PLATFORM_NAME`` is exported into the worker environment.

``jax_platform``
   String (``cpu`` or ``gpu``) exported as ``JAX_PLATFORM_NAME`` before JAX is imported. Defaults to ``cpu``.

   Used in ``_01b_run_survey_simulation.py``, ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py``.

``parallel_systems``
   (Optional; only consumed by ``_04_jax_create_synthetic_images.py``.) Boolean. When ``True``, systems are bucketed by ``lens_model_list`` signature so that the first system in a bucket pays the JIT-warmup cost and the rest reuse the cached compilation. Defaults to ``False``.

``batch_size``
   (Optional; only consumed by ``_04_jax_create_synthetic_images.py``.) Integer cap on bucket size when ``parallel_systems`` is enabled — useful as a memory ceiling on GPU. Defaults to ``8``.

``instrument``
--------------

Single string identifying the instrument: ``roman``, ``jwst``, or ``hwo``. Lower-cased and validated against the calling script's ``SUPPORTED_INSTRUMENTS`` by ``PipelineHelper``. Drives loading of the appropriate ``mejiro.instruments`` class (``Roman``, ``JWST``, or ``HWO``).

``survey``
----------

Parameters for the survey simulation (population draw, deflector/source cuts, source catalog wiring).

``runs``
   Number of independent simulation runs. Each run gets a unique random seed and processes one galaxy table.

   Stored as ``pipeline.runs`` by ``PipelineHelper``; used in ``_01b_run_survey_simulation.py`` to build the per-run task list (round-robining detectors and pre-computed galaxy tables).

``num_galaxy_tables``
   Number of independent galaxy-population tables generated by ``_01a_generate_galaxy_tables.py``. Higher values give more intrinsic diversity but cost more compute. Defaults to ``pipeline.runs`` when absent.

   Used in ``_01a_generate_galaxy_tables.py`` to build the task list of tables to generate.

``speed_factor``
   Integer ``>= 1`` passed to ``slsim``'s ``LensPop.draw_population`` to speed up population draws at the cost of completeness. Defaults to ``1``.

   Used in ``_01b_run_survey_simulation.py`` for both the total-population and detectable-population draws.

``area``
   Survey area in ``deg2`` (the unit is asserted in ``_01b_run_survey_simulation.py`` via ``Quantity(value=area, unit='deg2')``). Must match the ``fsky`` value of the chosen SkyPy config.

   Used in ``_01a_generate_galaxy_tables.py`` to build ``sky_area`` for the SkyPy and SLHammocks pipelines, and in ``_01b_run_survey_simulation.py`` for the same purpose plus the per-square-degree lens density logged after the survey.

``skypy_config``
   Name of the SkyPy configuration file (without extension) under ``mejiro/data/skypy/<skypy_config>/``. For Roman, ``_01a_generate_galaxy_tables.py`` appends the lower-cased SCA string to find the per-detector YAML.

   Used in ``_01a_generate_galaxy_tables.py`` to locate the SkyPy config that drives the galaxy-population draw.

``write_to_csv``
   Boolean. When ``True``, ``_01b_run_survey_simulation.py`` exports per-run population tables (``total_pop_<run_id>.csv`` and ``detectable_pop_<run_id>.csv``) via ``slsim_util.write_lens_population_to_csv``.

``total_population``
   Boolean. When ``True``, ``_01b_run_survey_simulation.py`` additionally draws the full (pre-cut) lens population, computes SNRs for every candidate, and (if ``write_to_csv`` is also ``True``) writes the result to CSV.

``use_real_sources``
   Boolean. When ``True``, ``_01b_run_survey_simulation.py`` constructs the source population with ``extended_source_type='catalog_source'`` and forwards ``catalog_source_kwargs`` to slsim's ``sources.Galaxies``.

``catalog_source_kwargs``
   Dict forwarded under the key ``extended_source_kwargs`` to ``slsim.Sources.Galaxies`` (and thence to ``slsim.Lenses.lens_pop.LensPop``). See the upstream slsim documentation for the accepted keys (the example configs use ``catalog_path``, ``catalog_type``, ``sersic_fallback``, ``max_scale``).

   Used in ``_01b_run_survey_simulation.py`` only when ``use_real_sources`` is ``True``.

``use_slhammocks_pipeline``
   Boolean. When ``True``, ``_01a_generate_galaxy_tables.py`` also runs ``SLHammocksPipeline`` to draw a dark-matter halo catalog (``halo_galaxies``), and ``_01b_run_survey_simulation.py`` reconstructs the deflector population from ``deflectors.CompoundLensHalosGalaxies`` rather than ``deflectors.AllLensGalaxies``.

``slhammocks_pipeline_kwargs``
   Dict of keyword arguments forwarded to ``slsim.Pipelines.sl_hammocks_pipeline.SLHammocksPipeline``. See the upstream slsim documentation for accepted keys.

   Used in ``_01a_generate_galaxy_tables.py``. Note that ``skypy_config`` inside this dict is rewritten in-place to an absolute path under ``mejiro/data/skypy/slhammocks/`` (for Roman, the per-detector variant is selected).

``detectors``
   List of detector IDs (for Roman, SCAs 1-18). Stored as ``pipeline.detectors`` by ``PipelineHelper`` and used to round-robin runs across detectors in ``_01a_generate_galaxy_tables.py`` and ``_01b_run_survey_simulation.py``. Typically referenced via a YAML anchor (``&detectors``) so the same list is reused under ``psf.detectors``.

``bands``
   List of photometric bands for which SkyPy computes magnitudes. These magnitudes are stored on the resulting strong-lens objects (``physical_params``).

   Used in ``_01b_run_survey_simulation.py`` for SNR-candidate construction and CSV export, and in ``_02_build_lens_list.py`` when calling ``GalaxyGalaxy.from_slsim(..., bands=bands)``.

``remap_bands``
   (Optional; only present in the rung-1 challenge configs.) Dict of ``{destination_band: source_band}`` pairs that override which catalog cutout backs each band in ``source_images``. Self-mappings are no-ops; omit a band to leave its default in place.

   Used in ``_02_build_lens_list.py``: when present, ``slsim_util.remap_source_images`` is called on each constructed lens. Defaults to ``None`` (no remapping).

``deflector_cut_band``, ``deflector_cut_band_max``
   Band and maximum magnitude used to cut the deflector population before lens drawing. Passed to slsim as ``kwargs_deflector_cut={'band': ..., 'band_max': ...}``.

   Used in ``_01b_run_survey_simulation.py``.

``deflector_z_min``, ``deflector_z_max``
   Minimum and maximum deflector redshift. Used in ``_01a_generate_galaxy_tables.py`` for the SLHammocks pipeline and in ``_01b_run_survey_simulation.py`` for ``kwargs_deflector_cut``.

``source_cut_band``, ``source_cut_band_max``
   Band and maximum magnitude used to cut the source population. Passed to slsim as ``kwargs_source_cut={'band': ..., 'band_max': ...}``.

   Used in ``_01b_run_survey_simulation.py``.

``source_z_min``, ``source_z_max``
   Minimum and maximum source redshift. Used in ``_01b_run_survey_simulation.py`` for ``kwargs_source_cut``.

``min_image_separation``, ``max_image_separation``
   Minimum and maximum image separation in arcseconds (units documented in the YAML inline comments). Combined into ``kwargs_lens_detectable_cut`` and passed to ``LensPop.draw_population`` in ``_01b_run_survey_simulation.py``.

``mag_arc_limit_band``, ``mag_arc_limit``
   Band and maximum arc magnitude. Combined as ``{mag_arc_limit_band: mag_arc_limit}`` and added to ``kwargs_lens_detectable_cut`` in ``_01b_run_survey_simulation.py``.

``magnification``
   Minimum total magnification required of a candidate. Lenses with ``strong_lens.physical_params['magnification']`` below this value are skipped before the (expensive) SNR check in ``_01b_run_survey_simulation.py``.

``subhalos``
------------

Dark-matter substructure parameters. Consumed by ``_03_generate_subhalos.py``.

``fraction``
   Fraction of detectable systems (in ``[0, 1]``) that receive a substructure realization. When ``< 1.0``, ``_03_generate_subhalos.py`` seeds an ``np.random`` mask with ``seed`` so the choice of which systems get substructure is reproducible.

``pyhalo_model``
   String name of the pyHalo preset model (e.g., ``CDM``). Resolved via ``pyHalo.preset_models.preset_model_from_name`` and called with ``realization_kwargs``.

   Used in ``_03_generate_subhalos.py``.

``realization_kwargs``
   Dict forwarded as ``**realization_kwargs`` to the pyHalo preset-model constructor. See the upstream pyHalo documentation for the accepted keys. The example configs include ``log_mlow``, ``log_mhigh``, ``LOS_normalization``, ``concentration_model_subhalos``, ``concentration_model_fieldhalos``, ``shmf_log_slope``, and (Roman-only) ``r_tidal``, ``sigma_sub`` (see Section 3.1 and Section 6.3 of `Gilman et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.6077G/abstract>`__).

   Used in ``_03_generate_subhalos.py``. If ``cone_opening_angle_arcsec`` is not present, ``_03_generate_subhalos.py`` injects it as ``lens.get_einstein_radius() * 3``. Also written verbatim as per-system HDF5 attributes in ``_06_h5_export.py`` (each key carries the description ``"See pyHalo documentation"``).

``synthetic_image``
-------------------

Parameters for rendering idealized (PSF-convolved, noise-free) images. Each value here is forwarded to ``mejiro.synthetic_image.SyntheticImage``.

``bands``
   List of photometric bands for which to render synthetic images. May be a subset of ``survey.bands``.

   Used in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, ``_04_jax_create_synthetic_images.py``, ``romanisim_pipeline.py``, and ``_06_h5_export.py``.

``fov_arcsec``
   Field of view in arcseconds (documented in the ``SyntheticImage`` docstring). Forwarded to ``SyntheticImage(fov_arcsec=...)``.

   Used in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py``.

``supersampling_compute_mode``
   String forwarded to lenstronomy's ``kwargs_numerics`` as ``compute_mode`` (e.g., ``adaptive``).

   Used in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py``.

``supersampling_factor``
   Integer supersampling factor forwarded to lenstronomy's ``kwargs_numerics``. Typically referenced via a YAML anchor (``&supersampling_factor``) so ``psf.oversamples`` can include the same value.

   Used in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, ``_04_jax_create_synthetic_images.py``, and (for HDF5 metadata) ``_06_h5_export.py``.

``pieces``
   Boolean forwarded to ``SyntheticImage(pieces=...)``. When ``True``, lens and source surface brightness are computed and stored separately.

   Used in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py``.

``serialization``
   String selecting the on-disk format for each ``SyntheticImage`` written by step 04. One of:

   - ``full`` (default): pickle the entire ``SyntheticImage`` (including the embedded ``StrongLens`` and any pyhalo realization). Required by the galsim path (``_05_galsim.py``) and by analysis scripts that need the full lens model (e.g., ``projects/roman_data_challenge/substructure_snr_histogram.py``).
   - ``lightweight``: write a compact ``.npz`` per (system, band) containing the image as ``float32`` plus a JSON metadata blob with only what the romanisim path consumes downstream (redshifts, Einstein radius, per-band magnitudes, detector position, ``magnitude_zeropoint``, etc.). Roughly 20× smaller per (system, band) than ``full``; incompatible with the galsim path. Loaded transparently by ``mejiro.utils.util.load_synthetic_image``, which auto-detects the extension and returns a :class:`mejiro.synthetic_image.LightweightSyntheticImage` shim.

   Used in ``_04_create_synthetic_images.py`` and ``_04_jax_create_synthetic_images.py`` (writers); in ``romanisim_pipeline.py``, ``_06_h5_export_romanisim.py``, ``projects/roman_data_challenge/rung_1.py``, and ``calculate_snrs.py`` (readers, via the unified loader). ``_05_galsim.py`` raises at startup when ``serialization == 'lightweight'`` because the galsim engine requires the full ``SyntheticImage``.

``exposure``
------------

(Roman only.) Romanisim observation metadata. Consumed by ``romanisim_pipeline.py``.

``ma_table_number``
   Integer Multi-Accumulation (MA) table number. Indexes ``romanisim.parameters.read_pattern`` to determine the read pattern (and hence the total exposure time, ``parameters.read_time * read_pattern[-1][-1]``); also written to ``meta['exposure']['ma_table_number']`` for the simulated observation.

``date``
   Observation date as an ISO-8601 string (e.g., ``2027-04-15T00:00:00``). Converted to ``astropy.time.Time`` and assigned to ``meta['exposure']['start_time']``.

``coordinates.ra``, ``coordinates.dec``
   Pointing right ascension and declination. ``romanisim_pipeline.py`` constructs ``SkyCoord(ra=ra * u.deg, dec=dec * u.deg)``, so the values are interpreted in degrees.

``imaging``
-----------

Parameters for the detector-effects step. Consumed by ``_05_galsim.py`` and ``calculate_snrs.py``.

``exposure_time``
   Exposure time in seconds (documented in the ``Exposure`` docstring). Typically referenced via a YAML anchor (``&exposure_time``) so ``snr.snr_exposure_time`` can reuse it.

   Used in ``_05_galsim.py`` to build each ``Exposure``.

``engine``
   String selecting the detector-effects engine, e.g., ``galsim`` (see :doc:`../mejiro/engines` for the available engines).

   Used in ``_05_galsim.py`` and in ``calculate_snrs.py`` for the SNR-rebuild path.

``engine_params``
   Dict of engine-specific parameters forwarded to the simulation engine. See :class:`mejiro.exposure.Exposure` and the engine modules under :mod:`mejiro.engines` for the accepted keys per engine. For the GalSim Roman engine, the example configs include ``rng_seed``, ``min_zodi_factor``, and boolean toggles ``sky_background``, ``detector_effects``, ``poisson_noise``, ``reciprocity_failure``, ``dark_noise``, ``nonlinearity``, ``ipc``, ``read_noise``.

   Used in ``_01b_run_survey_simulation.py`` for the SNR-detection ``Exposure``, in ``_05_galsim.py`` for the production ``Exposure``, and in ``calculate_snrs.py`` for the SNR-rebuild ``Exposure``.

``snr``
-------

Parameters for SNR-based detectability cuts (in ``_01b_run_survey_simulation.py``) and for the standalone SNR calculation (``calculate_snrs.py``).

``snr_band``
   Band used to render the SNR-evaluation ``SyntheticImage``.

   Used in ``_01b_run_survey_simulation.py``.

``snr_exposure_time``
   Exposure time in seconds for the SNR ``Exposure``. Typically a YAML reference (``*exposure_time``) to ``imaging.exposure_time``.

   Used in ``_01b_run_survey_simulation.py`` and ``calculate_snrs.py``.

``snr_fov_arcsec``
   Field of view in arcseconds for the SNR ``SyntheticImage``.

   Used in ``_01b_run_survey_simulation.py`` and ``calculate_snrs.py``.

``snr_supersampling_compute_mode``
   ``compute_mode`` for ``kwargs_numerics`` of the SNR ``SyntheticImage``.

   Used in ``_01b_run_survey_simulation.py`` and ``calculate_snrs.py``.

``snr_supersampling_factor``
   ``supersampling_factor`` for ``kwargs_numerics`` of the SNR ``SyntheticImage``. Typically referenced via a YAML anchor (``&snr_supersampling_factor``) so ``psf.oversamples`` can include the same value.

   Used in ``_01b_run_survey_simulation.py`` and ``calculate_snrs.py``.

``snr_threshold``
   Minimum total-system SNR for a candidate to be considered detectable. Applied in ``_01b_run_survey_simulation.py`` after the per-pixel SNR calculation.

``snr_per_pixel_threshold``
   Per-pixel SNR threshold passed to ``snr_calculation.get_snr`` and ``Exposure.get_snr``.

   Used in ``_01b_run_survey_simulation.py``, ``_06_h5_export.py``, and ``calculate_snrs.py``.

``snr_detector_position``
   (Optional.) Tuple ``[x, y]`` giving the detector position used when building the SNR PSF. Defaults to ``(2554, 2554)`` (the detector center) when absent.

   Used in ``_01b_run_survey_simulation.py``.

``psf``
-------

PSF cache and generation parameters. Consumed primarily by ``_00_cache_psfs.py`` (Roman) and read elsewhere when constructing ``kwargs_psf``.

``bands``
   List of bands for which PSFs are pre-computed.

   Used in ``_00_cache_psfs.py``.

``oversamples``
   List of integer oversampling factors for which PSFs are pre-computed. Typically the YAML references ``[*snr_supersampling_factor, *supersampling_factor]`` so both the SNR and synthetic-image paths find cached PSFs.

   Used in ``_00_cache_psfs.py``.

``num_pixes``
   List of PSF array sizes in pixels. **The first element is used by the pipeline** when constructing ``kwargs_psf`` (the additional elements are still pre-computed by ``_00_cache_psfs.py`` so that ad-hoc analysis scripts can request other sizes from the cache).

   Used in ``_00_cache_psfs.py`` to enumerate PSFs to generate, and in ``_01b_run_survey_simulation.py``, ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, ``_04_jax_create_synthetic_images.py``, ``calculate_snrs.py``, and ``_06_h5_export.py`` as ``num_pixes[0]`` for the active PSF.

``detectors``
   List of detector IDs for which PSFs are pre-computed. Typically references ``*detectors`` so it tracks ``survey.detectors``.

   Used in ``_00_cache_psfs.py`` and ``_06_h5_export.py``.

``divide_up_detector``
   Integer ``N`` controlling how each detector is divided into an ``N x N`` grid of PSF-evaluation positions (e.g., ``5`` -> 25 positions per detector). Roman-specific; ``psf_config.get('divide_up_detector')`` is ``None`` for HWO.

   Used in ``_00_cache_psfs.py`` (via ``roman_util.divide_up_sca``), in ``_04_create_synthetic_images.py``, ``_04_create_synthetic_images_interpol.py``, and ``_04_jax_create_synthetic_images.py`` to pick a random detector position per image, in ``_06_h5_export.py`` to enumerate positions for the PSF HDF5 dataset, and in ``romanisim_pipeline.py`` to define the PSF buckets that group tiles within each 4088x4088 detector image (asserted to divide both ``GRID_SIDE`` and ``4088``).

``dataset``
-----------

Output-dataset options. Consumed by ``_06_h5_export.py``.

``version``
   Dataset version string. Embedded in the HDF5 filename as ``<pipeline_label>_v_<version>.h5`` (with ``.`` replaced by ``_``) and written to the file-level attribute ``dataset_version``.

``labeled``
   (Challenge-only attribute, present in the ``roman_data_challenge_*`` configs.) Boolean indicating whether the exported dataset includes ground-truth labels. Read by downstream consumers of the Roman Data Challenge dataset; not consumed by the pipeline scripts in ``mejiro/pipeline/`` themselves.

``include_psfs``
   Boolean. When ``True``, ``_06_h5_export.py`` adds a ``psfs`` group to the HDF5 file containing cached PSFs for every (detector, position, band) combination.

``include_synthetic_images``
   Boolean. When ``True``, ``_06_h5_export.py`` writes the band-matched synthetic-image arrays alongside each ``Exposure`` dataset.
