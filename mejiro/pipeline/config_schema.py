"""The schema of mejiro's pipeline config: every key, its type, default and description.

This is the single source of truth for the YAML config. It's used to:

- validate a config when a pipeline step starts (:func:`validate`), so a bad config stops with
  a clear message instead of a ``KeyError`` deep inside a step;
- fill in defaults (:func:`apply_defaults`), so the pipeline scripts read keys directly;
- generate the key-by-key reference in ``docs/source/usage/configuration.rst`` at docs build
  time (see ``docs/source/_ext/config_reference.py``).

To add a key, add a :class:`Key` to its :class:`Section` in :data:`SCHEMA`. The tests in
``tests/test_docs/test_config_reference.py`` fail if the schema and the code disagree about
which steps read it.

This module only uses the standard library: Read the Docs imports it with none of mejiro's
dependencies installed, so keep ``yaml``, ``numpy`` and the like out of it. Key descriptions
are reStructuredText, since they're rendered into the docs.
"""
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Where the error message for an invalid config sends people with a v3.0.0 config. Update this
# when unreleased.md is renamed at a release; test_config_reference.py checks it.
UPGRADE_NOTES = 'docs/releases/unreleased.md'

# Step name (a script's SCRIPT_NAME, as passed to PipelineHelper) -> script filename. The last
# three aren't numbered steps, but they read config keys too.
STEPS = {
    '00': '_00_cache_psfs.py',
    '01a': '_01a_generate_galaxy_tables.py',
    '01b': '_01b_run_survey_simulation.py',
    '02': '_02_build_lens_list.py',
    '03': '_03_generate_subhalos.py',
    '04': '_04_create_synthetic_images.py',
    '05_galsim': '_05_galsim.py',
    '05_romanisim': '_05_romanisim.py',
    'snr': 'calculate_snrs.py',
    '06': '_06_h5_export.py',
    'compact_03': 'compact_03_realizations.py',
    'make_substructure_csv': 'make_substructure_csv.py',
    'pipeline': 'mejiro_pipeline.py',
}

# Stands for every step that goes through PipelineHelper, including project scripts with their
# own step names. PipelineHelper.__init__ reads these keys, so none of those steps can run
# without them. The standalone scripts read the config themselves, so it doesn't cover them.
EVERY_STEP = '*'
STANDALONE = ('compact_03', 'make_substructure_csv')

# A key is absent, as opposed to set to null.
_MISSING = object()
_NO_DEFAULT = object()


class ConfigError(ValueError):
    """A pipeline config that can't be used, with every problem found listed in the message."""


def is_romanisim_step(step_name):
    """Whether a step-05 variant name is the romanisim one.

    The step directory is the only reliable signal. ``imaging.engine`` only tells ``Pipeline``
    which step 05 to run, and running ``_05_romanisim.py`` directly ignores it, so the config
    can say ``'galsim'`` even when ``_05_romanisim`` produced the data. And the file extension
    is a serialization detail that both engines may share.
    """
    return step_name.startswith('05_romanisim')


def _lookup(config, dotted):
    """The value at a dotted key, or ``_MISSING`` if it's absent."""
    value = config
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


# -- Types

@dataclass(frozen=True)
class Type:
    """A value type: how it reads in docs and errors, and how to check it."""
    label: str
    check: Callable[[Any], bool]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)  # bool is a subclass of int


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _list_of(item, label, length=None):
    def check(value):
        return (isinstance(value, (list, tuple))
                and (length is None or len(value) == length)
                and all(item.check(element) for element in value))
    return Type(label, check)


STR = Type('string', lambda value: isinstance(value, str))
BOOL = Type('boolean', lambda value: isinstance(value, bool))
INT = Type('integer', _is_int)
NUMBER = Type('number', _is_number)
MAPPING = Type('mapping', lambda value: isinstance(value, dict))
LIST_OF_STR = _list_of(STR, 'list of strings')
LIST_OF_INT = _list_of(INT, 'list of integers')
# YAML loads an unquoted 2027-04-15T00:00:00 as a datetime, and an unquoted 0.1 as a float
DATE = Type('date or ISO-8601 string', lambda value: isinstance(value, (str, datetime.date)))
VERSION = Type('string or number', lambda value: isinstance(value, str) or _is_number(value))
POSITION = _list_of(NUMBER, 'list of two numbers', length=2)


# -- Conditions on a requirement

@dataclass(frozen=True)
class Context:
    """What a condition can look at: the config, and the step's command-line options."""
    config: dict
    prev_step: Optional[str]
    level: Optional[str]


@dataclass(frozen=True)
class Condition:
    """When a key is required. ``text`` is reStructuredText, and it's shown in docs and errors."""
    text: str
    applies: Callable[[Context], bool]


def _is_true(dotted):
    return lambda context: _lookup(context.config, dotted) is True


def _romanisim_input(context):
    return context.prev_step is not None and is_romanisim_step(context.prev_step)


def _partial_substructure(context):
    fraction = _lookup(context.config, 'subhalos.fraction')
    return _is_number(fraction) and fraction < 1


def _is_roman(context):
    instrument = _lookup(context.config, 'instrument')
    return isinstance(instrument, str) and instrument.lower() == 'roman'


USE_JAX = Condition('``jaxtronomy.use_jax`` is ``True``', _is_true('jaxtronomy.use_jax'))
USE_REAL_SOURCES = Condition('``survey.use_real_sources`` is ``True``', _is_true('survey.use_real_sources'))
USE_SLHAMMOCKS = Condition('``survey.use_slhammocks_pipeline`` is ``True``',
                           _is_true('survey.use_slhammocks_pipeline'))
LABELED = Condition('``dataset.labeled`` is ``True``', _is_true('dataset.labeled'))
PARTIAL_SUBSTRUCTURE = Condition('``subhalos.fraction`` is below 1', _partial_substructure)
ROMAN = Condition('``instrument`` is ``roman``', _is_roman)
ROMANISIM_INPUT = Condition('the input came from ``_05_romanisim.py``', _romanisim_input)
GALSIM_INPUT = Condition('the input came from ``_05_galsim.py``', lambda context: not _romanisim_input(context))
LEVEL_L2 = Condition('``--level`` is ``l2``', lambda context: context.level in (None, 'l2'))


@dataclass(frozen=True)
class Need:
    """A step that requires a key only under a condition."""
    step: str
    when: Condition


# -- The schema's building blocks

@dataclass(frozen=True)
class Key:
    """One config key.

    Parameters
    ----------
    name : str
        Dotted name, e.g. ``'subhalos.serialization'``.
    type : Type
        Its type. Pass-through dicts, whose contents belong to another package, are ``MAPPING``
        and aren't looked inside.
    description : str
        reStructuredText for the configuration reference.
    required_by : tuple
        Steps that can't run without it: step names, :data:`EVERY_STEP`, or a :class:`Need` for
        a step that requires it only sometimes.
    read_by : tuple
        Steps that read it but don't require it, because it has a default.
    default : optional
        Filled in by :func:`apply_defaults` when the key is absent.
    nullable : bool
        Whether ``null`` is a valid value.
    choices : tuple
        The allowed values, if they're limited.
    runtime : bool
        Set by the pipeline as it runs, so it can't appear in a config file.
    """
    name: str
    type: Type
    description: str
    required_by: tuple = ()
    read_by: tuple = ()
    default: Any = _NO_DEFAULT
    nullable: bool = False
    choices: tuple = ()
    runtime: bool = False

    @property
    def has_default(self):
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class Section:
    """A group of keys, which is a section of the configuration reference.

    ``name`` is the YAML mapping the keys sit under, or ``''`` for top-level keys. ``required_by``
    lists steps that read the mapping itself, whatever keys are in it. ``outro`` is text after the
    keys; ``{instrument_support}`` in it becomes the list of instruments each step supports.
    """
    title: str
    intro: str
    keys: tuple
    name: str = ''
    required_by: tuple = ()
    outro: str = ''


def _core(step, script, extra=''):
    return Key(f'cores.script_{step}', INT, f'Workers for ``{script}``.{extra}', required_by=(step,))


# -- The schema

SCHEMA = (
    Section(
        title='Global attributes',
        intro='These keys sit at the top level of the YAML file.',
        keys=(
            Key('data_dir', STR, nullable=True,
                required_by=(EVERY_STEP, 'compact_03', 'make_substructure_csv'), read_by=('pipeline',),
                description="""
                Base directory under which all pipeline output is written. If ``null``, the
                ``--data_dir`` CLI argument must be supplied to override it, and a step stops
                with an error if neither gives a value. The per-pipeline output directory is
                ``<data_dir>/<pipeline_label>``.

                Used by ``PipelineHelper`` to set ``pipeline_dir`` for every pipeline script.
                """),
            Key('pipeline_label', STR,
                required_by=(EVERY_STEP, 'compact_03', 'make_substructure_csv'), read_by=('pipeline',),
                description="""
                Name of the pipeline run. Used to construct the output directory
                (``<data_dir>/<pipeline_label>/``) and as a prefix for output filenames. When
                ``dev: True``, ``_dev`` is appended.
                """),
            Key('psf_cache_dir', STR, nullable=True, required_by=(EVERY_STEP,),
                description="""
                Directory where cached STPSF PSFs are stored as ``.npy`` files. May be an
                absolute path, a relative path, or ``null``. A relative path is resolved against
                ``data_dir``; if that directory doesn't exist, ``PipelineHelper`` falls back to
                resolving it against the repository root (this is how ``simple.yaml``'s
                ``tests/test_data`` works). When ``null``, ``PipelineHelper`` defaults to
                ``mejiro/data/psfs/<instrument>``.

                Used in ``_00_cache_psfs.py`` as the write target, and in
                ``_01b_run_survey_simulation.py``, ``_04_create_synthetic_images.py``, and
                ``calculate_snrs.py`` as the read target when constructing ``kwargs_psf``.
                """),
            Key('dev', BOOL,
                required_by=(EVERY_STEP, 'compact_03', 'make_substructure_csv'), read_by=('pipeline',),
                description="""
                Development-mode flag. When ``True``, ``_dev`` is appended to ``pipeline_label``
                so dev runs do not overwrite production output.

                Used by ``PipelineHelper`` to set the pipeline directory.
                """),
            Key('nice', INT, required_by=(EVERY_STEP,),
                description="""
                Process ``nice`` value applied to every pipeline worker. Set via ``os.nice(...)``
                in ``PipelineHelper.__init__``.
                """),
            Key('show_progress_bar', BOOL, required_by=(EVERY_STEP,),
                description="""
                When ``True``, tqdm bars are shown during long inner loops (e.g., the per-run
                candidate loops in ``_01b_run_survey_simulation.py``, where it toggles the
                per-candidate progress bars).
                """),
            Key('logging_level', STR, default='INFO',
                read_by=(EVERY_STEP, 'compact_03', 'make_substructure_csv'),
                description="""
                Sets the root logger's level (e.g., ``INFO``, ``WARNING``, ``DEBUG``). All
                pipeline console output goes through ``logging``, and Python warnings are
                captured onto the ``py.warnings`` logger at ``WARNING``, so this one key controls
                what reaches the console. To hide third-party warnings, use ``ERROR``.

                Applied by ``PipelineHelper.configure_logging_and_warnings``, called from
                ``PipelineHelper.__init__`` and from the spawn-worker initializers, and directly
                by the standalone ``compact_03_realizations.py`` and ``make_substructure_csv.py``.
                The following deliberately bypass it:

                - tqdm progress bars write directly to stderr.
                - ``util.print_execution_time`` logs at level 100, so the execution time always
                  prints.
                - ``_00_cache_psfs.py`` sets ``mejiro.engines.stpsf_engine`` to ``ERROR``.
                - The JAX-enabled steps set ``jax._src.xla_bridge`` to ``CRITICAL``.
                - ``_05_romanisim.py`` drops sub-``ERROR`` records from romancal, romanisim, CRDS,
                  and ``py.warnings``.
                """),
            Key('limit', INT, nullable=True, required_by=(EVERY_STEP, 'make_substructure_csv'),
                description="""
                Maximum number of systems each script should process, or ``null`` for no limit.

                Used in ``_01b_run_survey_simulation.py`` to cap the number of detectable lenses
                per run, in ``_02_build_lens_list.py`` to short-circuit the lens-conversion loop,
                in ``_03_generate_subhalos.py``, ``_04_create_synthetic_images.py``,
                ``_05_galsim.py``, and ``calculate_snrs.py`` to subsample input pickles
                (sequentially with ``--sequential``, otherwise via ``np.random.choice``), and in
                ``_05_romanisim.py`` to subsample lens IDs per SCA.
                """),
            Key('seed', INT, required_by=('01a', '01b', Need('03', PARTIAL_SUBSTRUCTURE), '04', '05_romanisim'),
                description="""
                Global random seed.

                Used in ``_01a_generate_galaxy_tables.py`` to seed the per-table draw via
                ``hash((seed, 'table', table_index))``, in ``_01b_run_survey_simulation.py`` for
                the per-run draw via ``hash((seed, run))``, in ``_03_generate_subhalos.py`` to
                seed the substructure assignment mask, in ``_04_create_synthetic_images.py`` to
                choose each system's detector position and whether it is rendered deflector-only
                (see ``synthetic_image.deflector_only_fraction``), and in ``_05_romanisim.py`` to
                derive per-batch RNGs
                (``batch_seed = seed + sca_num * 10000 + band_idx * 1000 + batch_idx``). Typically
                also referenced from ``imaging.engine_params.rng_seed`` via a YAML anchor.
                """),
        ),
    ),
    Section(
        title='``cores``',
        name='cores',
        intro="""
        Per-script worker counts for ``ProcessPoolExecutor``. Each key maps the script name to its
        worker count; ``PipelineHelper.calculate_process_count`` reads ``cores.script_<script_name>``.
        """,
        keys=(
            _core('00', '_00_cache_psfs.py'),
            _core('01a', '_01a_generate_galaxy_tables.py'),
            _core('01b', '_01b_run_survey_simulation.py'),
            _core('03', '_03_generate_subhalos.py'),
            _core('04', '_04_create_synthetic_images.py'),
            _core('05_galsim', '_05_galsim.py'),
            _core('05_romanisim', '_05_romanisim.py',
                  " Read directly as ``config['cores']['script_05_romanisim']``; the per-worker "
                  "thread count is derived as ``max(2, 64 // num_workers)``."),
            _core('snr', 'calculate_snrs.py'),
        ),
    ),
    Section(
        title='``jaxtronomy``',
        name='jaxtronomy',
        intro='Controls JAX-based acceleration for ray-shooting and image rendering.',
        keys=(
            Key('jaxtronomy.use_jax', BOOL, required_by=('01b', '02', '03', '04', 'snr'),
                description="""
                When ``True``, lens objects are built with ``use_jax=True`` so that ``jaxtronomy``
                is used in place of ``lenstronomy`` for ray-shooting. In
                ``_04_create_synthetic_images.py`` it also gates whether ``JAX_PLATFORM_NAME`` is
                exported into the worker environment.
                """),
            Key('jaxtronomy.jax_platform', STR,
                required_by=('01b', Need('03', USE_JAX), Need('04', USE_JAX), Need('snr', USE_JAX)),
                description="""
                ``cpu`` or ``gpu``, exported as ``JAX_PLATFORM_NAME`` before JAX is imported.
                """),
        ),
    ),
    Section(
        title='``instrument``',
        intro="""
        Drives loading of the appropriate ``mejiro.instruments`` class (``Roman``, ``JWST``,
        ``LSST``, or ``HWO``).
        """,
        keys=(
            Key('instrument', STR, required_by=(EVERY_STEP,),
                description="""
                Identifies the instrument: ``roman``, ``jwst``, ``lsst``, or ``hwo``. Lower-cased
                and validated against the calling script's ``SUPPORTED_INSTRUMENTS`` by
                ``PipelineHelper``.
                """),
        ),
        outro="""
        Not every step supports every instrument. ``PipelineHelper`` raises ``ValueError`` when a
        step is run with an instrument it doesn't support:

        {instrument_support}

        So ``Pipeline.run()`` only works end to end for Roman: it runs step 00 (unless in test
        mode) and always finishes with step 06. For JWST, LSST and HWO, run the supported steps
        individually. JWST and LSST have no step 03, so run step 04 with ``--prev-step 02`` to
        read the lenses straight from step 02.
        """,
    ),
    Section(
        title='``survey``',
        name='survey',
        intro="""
        Parameters for the survey simulation (population draw, deflector/source cuts, source
        catalog wiring).
        """,
        keys=(
            Key('survey.runs', INT, required_by=(EVERY_STEP,),
                description="""
                Number of independent simulation runs. Each run gets a unique random seed and
                processes one galaxy table.

                Stored as ``pipeline.runs`` by ``PipelineHelper``; used in
                ``_01b_run_survey_simulation.py`` to build the per-run task list (round-robining
                detectors and pre-computed galaxy tables).
                """),
            Key('survey.num_galaxy_tables', INT, required_by=('01a',),
                description="""
                Number of independent galaxy-population tables generated by
                ``_01a_generate_galaxy_tables.py``. Higher values give more intrinsic diversity
                but cost more compute.
                """),
            Key('survey.speed_factor', INT, required_by=('01b',),
                description="""
                Integer ``>= 1`` passed to ``slsim``'s ``LensPop.draw_population`` to speed up
                population draws at the cost of completeness. Used for both the total-population
                and detectable-population draws.
                """),
            Key('survey.area', NUMBER, required_by=('01a', '01b'),
                description="""
                Survey area in ``deg2`` (the unit is asserted in
                ``_01b_run_survey_simulation.py`` via ``Quantity(value=area, unit='deg2')``). Must
                match the ``fsky`` value of the chosen SkyPy config.

                Used in ``_01a_generate_galaxy_tables.py`` to build ``sky_area`` for the SkyPy and
                SLHammocks pipelines, and in ``_01b_run_survey_simulation.py`` for the same
                purpose plus the per-square-degree lens density logged after the survey.
                """),
            Key('survey.skypy_config', STR, required_by=('01a',),
                description="""
                Name of the SkyPy configuration file (without extension) under
                ``mejiro/data/skypy/<skypy_config>/``, which drives the galaxy-population draw.
                For Roman, ``_01a_generate_galaxy_tables.py`` appends the lower-cased SCA string
                to find the per-detector YAML.
                """),
            Key('survey.write_to_csv', BOOL, required_by=('01b',),
                description="""
                When ``True``, ``_01b_run_survey_simulation.py`` exports per-run population tables
                (``total_pop_<run_id>.csv`` and ``detectable_pop_<run_id>.csv``) via
                ``slsim_util.write_lens_population_to_csv``.
                """),
            Key('survey.total_population', BOOL, required_by=('01b',),
                description="""
                When ``True``, ``_01b_run_survey_simulation.py`` additionally draws the full
                (pre-cut) lens population, computes SNRs for every candidate, and (if
                ``write_to_csv`` is also ``True``) writes the result to CSV.
                """),
            Key('survey.use_real_sources', BOOL, required_by=('01b',),
                description="""
                When ``True``, ``_01b_run_survey_simulation.py`` constructs the source population
                with ``extended_source_type='catalog_source'`` and forwards
                ``catalog_source_kwargs`` to slsim's ``sources.Galaxies``.
                """),
            Key('survey.catalog_source_kwargs', MAPPING, required_by=(Need('01b', USE_REAL_SOURCES),),
                description="""
                Dict forwarded under the key ``extended_source_kwargs`` to
                ``slsim.Sources.Galaxies`` (and thence to ``slsim.Lenses.lens_pop.LensPop``). See
                the upstream slsim documentation for the accepted keys (the example configs use
                ``catalog_path``, ``catalog_type``, ``sersic_fallback``, ``max_scale``).
                """),
            Key('survey.use_slhammocks_pipeline', BOOL, required_by=('01a', '01b'),
                description="""
                When ``True``, ``_01a_generate_galaxy_tables.py`` also runs ``SLHammocksPipeline``
                to draw a dark-matter halo catalog (``halo_galaxies``), and
                ``_01b_run_survey_simulation.py`` reconstructs the deflector population from
                ``CompoundLensHalosGalaxies`` rather than ``GalaxyDeflectors``.
                """),
            Key('survey.slhammocks_pipeline_kwargs', MAPPING, required_by=(Need('01a', USE_SLHAMMOCKS),),
                description="""
                Dict of keyword arguments forwarded to
                ``slsim.Pipelines.sl_hammocks_pipeline.SLHammocksPipeline``. See the upstream
                slsim documentation for accepted keys. Note that ``skypy_config`` inside this dict
                is rewritten in-place to an absolute path under ``mejiro/data/skypy/slhammocks/``
                (for Roman, the per-detector variant is selected).
                """),
            Key('survey.detectors', LIST_OF_INT, required_by=(EVERY_STEP,),
                description="""
                List of detector IDs (for Roman, SCAs 1-18). Stored as ``pipeline.detectors`` by
                ``PipelineHelper`` and used to round-robin runs across detectors in
                ``_01a_generate_galaxy_tables.py`` and ``_01b_run_survey_simulation.py``. Typically
                referenced via a YAML anchor (``&detectors``) so the same list is reused under
                ``psf.detectors``.
                """),
            Key('survey.bands', LIST_OF_STR, required_by=('01b', '02'),
                description="""
                List of photometric bands for which SkyPy computes magnitudes. These magnitudes
                are stored on the resulting strong-lens objects (``physical_params``).

                Used in ``_01b_run_survey_simulation.py`` for SNR-candidate construction and CSV
                export, and in ``_02_build_lens_list.py`` when calling
                ``GalaxyGalaxy.from_slsim(..., bands=bands)``.
                """),
            Key('survey.remap_bands', MAPPING, nullable=True, default=None, read_by=('02',),
                description="""
                Dict of ``{destination_band: source_band}`` pairs that override which catalog
                cutout backs each band in ``source_images``. Self-mappings are no-ops; omit a band
                to leave its default in place. Required when lenses have catalog source images
                (``source_images`` in their parameters, e.g. with ``use_real_sources: True``);
                otherwise it can be left out.

                Whether it's needed depends on the lenses, so config validation can't enforce it:
                step 02 logs a warning at startup if ``use_real_sources`` is ``True`` and this is
                unset. ``_02_build_lens_list.py`` calls ``slsim_util.remap_source_images`` on each
                lens that has catalog source images, and raises ``ValueError`` if such a lens
                appears while ``remap_bands`` is unset or empty.
                """),
            Key('survey.deflector_cut_band', STR, required_by=('01b',),
                description="""
                Band used to cut the deflector population before lens drawing. Passed to slsim as
                ``kwargs_deflector_cut={'band': ..., 'band_max': ...}``.
                """),
            Key('survey.deflector_cut_band_max', NUMBER, required_by=('01b',),
                description="""
                Maximum magnitude used to cut the deflector population before lens drawing. Passed
                to slsim as ``kwargs_deflector_cut={'band': ..., 'band_max': ...}``.
                """),
            Key('survey.deflector_z_min', NUMBER, required_by=(Need('01a', USE_SLHAMMOCKS), '01b'),
                description="""
                Minimum deflector redshift. Used in ``_01a_generate_galaxy_tables.py`` for the
                SLHammocks pipeline and in ``_01b_run_survey_simulation.py`` for
                ``kwargs_deflector_cut``.
                """),
            Key('survey.deflector_z_max', NUMBER, required_by=(Need('01a', USE_SLHAMMOCKS), '01b'),
                description="""
                Maximum deflector redshift. Used in ``_01a_generate_galaxy_tables.py`` for the
                SLHammocks pipeline and in ``_01b_run_survey_simulation.py`` for
                ``kwargs_deflector_cut``.
                """),
            Key('survey.source_cut_band', STR, required_by=('01b',),
                description="""
                Band used to cut the source population. Passed to slsim as
                ``kwargs_source_cut={'band': ..., 'band_max': ...}``.
                """),
            Key('survey.source_cut_band_max', NUMBER, required_by=('01b',),
                description="""
                Maximum magnitude used to cut the source population. Passed to slsim as
                ``kwargs_source_cut={'band': ..., 'band_max': ...}``.
                """),
            Key('survey.source_z_min', NUMBER, required_by=('01b',),
                description='Minimum source redshift, used for ``kwargs_source_cut``.'),
            Key('survey.source_z_max', NUMBER, required_by=('01b',),
                description='Maximum source redshift, used for ``kwargs_source_cut``.'),
            Key('survey.min_image_separation', NUMBER, required_by=('01b',),
                description="""
                Minimum image separation in arcseconds. Combined into
                ``kwargs_lens_detectable_cut`` and passed to ``LensPop.draw_population`` in
                ``_01b_run_survey_simulation.py``.
                """),
            Key('survey.max_image_separation', NUMBER, required_by=('01b',),
                description="""
                Maximum image separation in arcseconds. Combined into
                ``kwargs_lens_detectable_cut`` and passed to ``LensPop.draw_population`` in
                ``_01b_run_survey_simulation.py``.
                """),
            Key('survey.mag_arc_limit_band', STR, required_by=('01b',),
                description="""
                Band of the arc magnitude limit. Combined as ``{mag_arc_limit_band: mag_arc_limit}``
                and added to ``kwargs_lens_detectable_cut`` in ``_01b_run_survey_simulation.py``.
                """),
            Key('survey.mag_arc_limit', NUMBER, required_by=('01b',),
                description="""
                Maximum arc magnitude. Combined as ``{mag_arc_limit_band: mag_arc_limit}`` and
                added to ``kwargs_lens_detectable_cut`` in ``_01b_run_survey_simulation.py``.
                """),
            Key('survey.magnification', NUMBER, required_by=('01b',),
                description="""
                Minimum total magnification required of a candidate. Lenses with
                ``strong_lens.physical_params['magnification']`` below this value are skipped
                before the (expensive) SNR check in ``_01b_run_survey_simulation.py``.
                """),
            Key('survey.cosmo', MAPPING, runtime=True, read_by=('01a', '01b'),
                description="""
                Set by steps 01a and 01b as they run, to the default ``astropy`` cosmology. Don't
                put it in a config file: validation rejects it.
                """),
        ),
    ),
    Section(
        title='``subhalos``',
        name='subhalos',
        required_by=('06',),
        intro="""
        Dark-matter substructure parameters. Consumed by ``_03_generate_subhalos.py``.
        """,
        keys=(
            Key('subhalos.fraction', NUMBER, required_by=('03',),
                description="""
                Fraction of detectable systems (in ``[0, 1]``) that receive a substructure
                realization. When ``< 1.0``, ``_03_generate_subhalos.py`` seeds an ``np.random``
                mask with ``seed`` so the choice of which systems get substructure is reproducible.
                """),
            Key('subhalos.pyhalo_model', STR, required_by=('03',),
                description="""
                Name of the pyHalo preset model (e.g., ``CDM``). Resolved via
                ``pyHalo.preset_models.preset_model_from_name`` and called with
                ``realization_kwargs``.
                """),
            Key('subhalos.realization_kwargs', MAPPING, required_by=('03', Need('06', LABELED)),
                description="""
                Dict forwarded as ``**realization_kwargs`` to the pyHalo preset-model constructor.
                See the upstream pyHalo documentation for the accepted keys. The example configs
                include ``log_mlow``, ``log_mhigh``, ``LOS_normalization``,
                ``concentration_model_subhalos``, ``concentration_model_fieldhalos``,
                ``shmf_log_slope``, and (Roman-only) ``r_tidal``, ``sigma_sub`` (see Section 3.1
                and Section 6.3 of `Gilman et al. 2020
                <https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.6077G/abstract>`__).

                If ``cone_opening_angle_arcsec`` is not present, ``_03_generate_subhalos.py``
                injects it as ``lens.get_einstein_radius() * 3``. Also written verbatim as
                per-system HDF5 attributes in ``_06_h5_export.py`` for systems with substructure
                when ``dataset.labeled`` is ``True`` (each key carries the description ``"See
                pyHalo documentation"``).
                """),
            Key('subhalos.serialization', STR, choices=('full', 'lightweight'), required_by=('03',),
                description="""
                Selects what ``_03_generate_subhalos.py`` writes for each system that receives
                substructure:

                - ``full``: pickle the lens together with its pyHalo realization object.
                - ``lightweight``: drop the realization object before pickling, via
                  ``mejiro.analysis.lensing.strip_realization``. The subhalos are already built
                  into the lens model (``kwargs_lens``), which is all step 04 reads, and the
                  realization object is most of the size of a substructured lens pickle. The
                  substructure flag is kept, so downstream steps still know which systems have
                  substructure.

                Choose ``full`` if anything downstream needs the realization object itself, for
                example to inspect or plot individual subhalos; it can't be recovered from
                ``lightweight`` output. Existing ``full`` output can be compacted afterwards with
                ``compact_03_realizations.py``.
                """),
        ),
    ),
    Section(
        title='``synthetic_image``',
        name='synthetic_image',
        intro="""
        Parameters for rendering idealized (PSF-convolved, noise-free) images in
        ``_04_create_synthetic_images.py``. Most values here are forwarded to
        ``mejiro.synthetic_image.SyntheticImage``.
        """,
        keys=(
            Key('synthetic_image.bands', LIST_OF_STR, required_by=('04', '05_romanisim', '06'),
                read_by=('pipeline',),
                description="""
                List of photometric bands for which to render synthetic images. May be a subset of
                ``survey.bands``.
                """),
            Key('synthetic_image.fov_arcsec', NUMBER, required_by=('04', '05_romanisim'),
                description="""
                Field of view in arcseconds (documented in the ``SyntheticImage`` docstring).
                Forwarded to ``SyntheticImage(fov_arcsec=...)``, and used in ``_05_romanisim.py``
                to size the tiles placed on each detector.
                """),
            Key('synthetic_image.supersampling_compute_mode', STR, required_by=('04',),
                description="""
                Forwarded to lenstronomy's ``kwargs_numerics`` as ``compute_mode`` (e.g.,
                ``adaptive``).
                """),
            Key('synthetic_image.supersampling_factor', INT, required_by=('04',),
                description="""
                Supersampling factor forwarded to lenstronomy's ``kwargs_numerics``. Typically
                referenced via a YAML anchor (``&supersampling_factor``) so ``psf.oversamples``
                can include the same value.
                """),
            Key('synthetic_image.pieces', BOOL, required_by=('04',),
                description="""
                Forwarded to ``SyntheticImage(pieces=...)``. When ``True``, lens and source
                surface brightness are computed and stored separately.
                """),
            Key('synthetic_image.serialization', STR, choices=('full', 'lightweight'),
                required_by=('04', '05_galsim'),
                description="""
                Selects the on-disk format for each ``SyntheticImage`` written by step 04:

                - ``full``: pickle the entire ``SyntheticImage`` (``.pkl``), including the
                  embedded ``StrongLens`` and any pyHalo realization. Required by the galsim path
                  (``_05_galsim.py``) and by analysis that needs the full lens model.
                - ``lightweight``: write a compact ``.npz`` per (system, band) containing the
                  image as ``float32`` plus a JSON metadata blob with only what the romanisim path
                  consumes downstream (redshifts, Einstein radius, per-band magnitudes, detector
                  position, ``magnitude_zeropoint``, etc.). Roughly 20× smaller per (system, band)
                  than ``full``; incompatible with the galsim path. Loaded transparently by
                  ``mejiro.synthetic_image.load_synthetic_image``, which auto-detects the extension
                  and returns a :class:`mejiro.synthetic_image.LightweightSyntheticImage` shim.

                Read back by ``_05_romanisim.py``, ``calculate_snrs.py``, and ``_06_h5_export.py``
                through the unified loader. ``_05_galsim.py`` raises at startup when
                ``serialization == 'lightweight'`` because the galsim engine requires the full
                ``SyntheticImage``.
                """),
            Key('synthetic_image.oversample', INT, default=1, read_by=('04', '05_galsim'),
                description="""
                When greater than ``1``, step 04 renders each image on a grid ``oversample`` times
                finer than the detector pixels, with no detector pixel integral applied.
                ``_05_romanisim.py`` bins these images to detector pixels at each dither's
                sub-pixel phase; ``_05_galsim.py`` needs detector-resolution input and raises if
                ``oversample`` is greater than ``1``. Step 04 then uses the cached PSF kernel at
                this oversampling as-is, so ``psf.oversamples`` must include the value (step 04
                raises if the kernel isn't cached).
                """),
            Key('synthetic_image.deflector_only_fraction', NUMBER, default=0.0, read_by=('04',),
                description="""
                Float in ``[0, 1]``. Fraction of systems rendered deflector-only: lens-galaxy
                light only, with no source and no lensing. The choice is made from ``seed`` and
                each system's name, so a system is deflector-only in every band and the choice
                stays the same across ``--resume``. The realized fraction is approximate.
                ``_06_h5_export.py`` records each system's ``deflector_only`` flag: in the HDF5
                file when ``dataset.labeled`` is ``True``, otherwise in the answer key.
                """),
        ),
    ),
    Section(
        title='``exposure``',
        name='exposure',
        intro="""
        (Roman only.) Romanisim observation metadata. Consumed by ``_05_romanisim.py``.
        ``calculate_snrs.py`` and ``_06_h5_export.py`` also read ``ma_table_number`` to work out
        the exposure time of romanisim output.
        """,
        keys=(
            Key('exposure.ma_table_number', INT,
                required_by=('05_romanisim', Need('snr', ROMANISIM_INPUT), Need('06', ROMANISIM_INPUT)),
                description="""
                Multi-Accumulation (MA) table number. Indexes ``romanisim.parameters.read_pattern``
                to determine the read pattern (and hence the total exposure time,
                ``parameters.read_time * read_pattern[-1][-1]``); also written to
                ``meta['exposure']['ma_table_number']`` for the simulated observation.
                """),
            Key('exposure.date', DATE, required_by=('05_romanisim',),
                description="""
                Observation date as an ISO-8601 value (e.g., ``2027-04-15T00:00:00``), which YAML
                loads as a date unless it's quoted. Converted to ``astropy.time.Time`` and
                assigned to ``meta['exposure']['start_time']``.
                """),
            Key('exposure.coordinates.ra', NUMBER, required_by=('05_romanisim',),
                description="""
                Pointing right ascension. ``_05_romanisim.py`` constructs
                ``SkyCoord(ra=ra * u.deg, dec=dec * u.deg)``, so the value is interpreted in
                degrees.
                """),
            Key('exposure.coordinates.dec', NUMBER, required_by=('05_romanisim',),
                description="""
                Pointing declination. ``_05_romanisim.py`` constructs
                ``SkyCoord(ra=ra * u.deg, dec=dec * u.deg)``, so the value is interpreted in
                degrees.
                """),
        ),
    ),
    Section(
        title='``imaging``',
        name='imaging',
        intro="""
        Parameters for the detector-effects step. Consumed by ``_05_galsim.py``; also read by
        ``_01b_run_survey_simulation.py``, ``calculate_snrs.py``, and ``_06_h5_export.py``.
        """,
        keys=(
            Key('imaging.exposure_time', NUMBER, required_by=('05_galsim', Need('06', GALSIM_INPUT)),
                description="""
                Exposure time in seconds (documented in the ``Exposure`` docstring). Typically
                referenced via a YAML anchor (``&exposure_time``) so ``snr.snr_exposure_time`` can
                reuse it.

                Used in ``_05_galsim.py`` to build each ``Exposure``, and in ``_06_h5_export.py``
                as the recorded exposure time of galsim exposures.
                """),
            Key('imaging.engine', STR, choices=('galsim', 'romanisim'), required_by=('05_galsim', 'pipeline'),
                description="""
                Selects the step 05 simulator (see :doc:`../mejiro/engines`).

                Used in ``Pipeline`` to choose between ``_05_galsim.py`` and ``_05_romanisim.py``,
                and in ``_05_galsim.py`` as the ``Exposure`` engine, so run that script only with
                ``galsim``. ``_05_romanisim.py`` doesn't read it when run directly. The SNR
                exposures in ``_01b_run_survey_simulation.py`` and ``calculate_snrs.py`` always
                use GalSim, whatever this is set to.
                """),
            Key('imaging.engine_params', MAPPING, required_by=('01b', '05_galsim', 'snr'),
                description="""
                Dict of engine-specific parameters forwarded to the simulation engine. See
                :class:`mejiro.exposure.Exposure` and the engine modules under
                :mod:`mejiro.engines` for the accepted keys per engine. For the GalSim Roman
                engine, the example configs include ``rng_seed``, ``min_zodi_factor``, and boolean
                toggles ``sky_background``, ``detector_effects``, ``poisson_noise``,
                ``reciprocity_failure``, ``dark_noise``, ``nonlinearity``, ``ipc``, ``read_noise``.

                Used in ``_01b_run_survey_simulation.py`` for the SNR-detection ``Exposure``, in
                ``_05_galsim.py`` for the production ``Exposure``, and in ``calculate_snrs.py``
                for the SNR-rebuild ``Exposure``. The SNR exposures always use GalSim and
                ``_05_romanisim.py`` doesn't read this key, so it must hold GalSim parameters even
                when ``engine`` is ``romanisim``.
                """),
            Key('imaging.serialization', STR, choices=('full', 'lightweight'),
                required_by=('05_galsim', 'snr', '06'),
                description="""
                Selects how ``_05_galsim.py`` writes each ``Exposure``:

                - ``full``: pickle the whole ``Exposure`` object (``.pkl``).
                - ``lightweight``: write a compact ``.npz`` containing the image data, any pieces,
                  and JSON metadata, via ``Exposure.save_lightweight``.

                This is separate from ``synthetic_image.serialization``, which controls step 04's
                output. ``_05_romanisim.py`` ignores it and always writes ``.npy`` cutouts.

                ``calculate_snrs.py`` and ``_06_h5_export.py`` use it to find galsim exposures by
                file extension (``PipelineHelper.exposure_extension``).
                """),
        ),
    ),
    Section(
        title='``snr``',
        name='snr',
        intro="""
        Parameters for SNR-based detectability cuts (in ``_01b_run_survey_simulation.py``) and for
        the standalone SNR calculation (``calculate_snrs.py``).
        """,
        keys=(
            Key('snr.snr_band', STR, required_by=('01b',),
                description='Band used to render the SNR-evaluation ``SyntheticImage``.'),
            Key('snr.snr_exposure_time', NUMBER, required_by=('01b', Need('snr', GALSIM_INPUT)),
                description="""
                Exposure time in seconds for the SNR ``Exposure``. Typically a YAML reference
                (``*exposure_time``) to ``imaging.exposure_time``. For romanisim input,
                ``calculate_snrs.py`` uses that output's real exposure time instead.
                """),
            Key('snr.snr_fov_arcsec', NUMBER, required_by=('01b', 'snr'),
                description='Field of view in arcseconds for the SNR ``SyntheticImage``.'),
            Key('snr.snr_supersampling_compute_mode', STR, required_by=('01b', 'snr'),
                description='``compute_mode`` for ``kwargs_numerics`` of the SNR ``SyntheticImage``.'),
            Key('snr.snr_supersampling_factor', INT, required_by=('01b', 'snr'),
                description="""
                ``supersampling_factor`` for ``kwargs_numerics`` of the SNR ``SyntheticImage``.
                Typically referenced via a YAML anchor (``&snr_supersampling_factor``) so
                ``psf.oversamples`` can include the same value.
                """),
            Key('snr.snr_threshold', NUMBER, required_by=('01b',),
                description="""
                Minimum total-system SNR for a candidate to be considered detectable. Applied in
                ``_01b_run_survey_simulation.py`` after the per-pixel SNR calculation.
                """),
            Key('snr.snr_per_pixel_threshold', NUMBER, required_by=('01b', 'snr'),
                description="""
                Per-pixel SNR threshold passed to ``snr_calculation.get_snr`` and
                ``Exposure.get_snr``.
                """),
            Key('snr.snr_detector_position', POSITION, default=(2043, 2043), read_by=('01b',),
                description="""
                Tuple ``[x, y]`` giving the detector position used when building the SNR PSF. The
                default is the detector center. Choose one of the positions step 00 caches PSFs at
                (see ``psf.divide_up_detector``; ``1`` gives the detector center), otherwise step
                01b computes the PSF with STPSF, which needs STPSF's data files.
                """),
        ),
    ),
    Section(
        title='``psf``',
        name='psf',
        intro="""
        PSF cache and generation parameters. Consumed primarily by ``_00_cache_psfs.py`` (Roman)
        and read elsewhere when constructing ``kwargs_psf``.
        """,
        keys=(
            Key('psf.bands', LIST_OF_STR, required_by=('00',),
                description='List of bands for which PSFs are pre-computed.'),
            Key('psf.oversamples', LIST_OF_INT, required_by=('00',),
                description="""
                List of integer oversampling factors for which PSFs are pre-computed. Typically
                the YAML references ``[*snr_supersampling_factor, *supersampling_factor]`` so both
                the SNR and synthetic-image paths find cached PSFs.
                """),
            Key('psf.num_pixes', LIST_OF_INT, required_by=('00', '01b', '04', 'snr'),
                description="""
                List of PSF array sizes in pixels. **The first element is used by the pipeline**
                when constructing ``kwargs_psf`` (the additional elements are still pre-computed
                by ``_00_cache_psfs.py`` so that ad-hoc analysis scripts can request other sizes
                from the cache).

                Used in ``_00_cache_psfs.py`` to enumerate PSFs to generate, and in
                ``_01b_run_survey_simulation.py``, ``_04_create_synthetic_images.py``, and
                ``calculate_snrs.py`` as ``num_pixes[0]`` for the active PSF.
                """),
            Key('psf.detectors', LIST_OF_INT, required_by=('00',),
                description="""
                List of detector IDs for which PSFs are pre-computed. Typically references
                ``*detectors`` so it tracks ``survey.detectors``.
                """),
            Key('psf.divide_up_detector', INT,
                required_by=('00', Need('04', ROMAN), Need('05_romanisim', LEVEL_L2)),
                description="""
                Integer ``N`` controlling how each detector is divided into an ``N x N`` grid of
                PSF-evaluation positions (e.g., ``4`` -> 16 positions per detector). ``N`` must
                divide the 4088-pixel detector evenly (1, 2, 4, 7, 8, 14, ...);
                ``roman_util.divide_up_sca`` raises otherwise. Roman-specific.

                Used in ``_00_cache_psfs.py`` (via ``roman_util.divide_up_sca``), in
                ``_04_create_synthetic_images.py`` to pick a detector position per image, and in
                ``_05_romanisim.py`` to define the PSF buckets that group tiles within each
                4088x4088 detector image (asserted to divide both ``GRID_SIDE`` and ``4088``).
                """),
        ),
    ),
    Section(
        title='``dataset``',
        name='dataset',
        intro='Output-dataset options. Consumed by ``_06_h5_export.py``.',
        keys=(
            Key('dataset.version', VERSION, required_by=('06',),
                description="""
                Dataset version. Embedded in the HDF5 filename as
                ``<pipeline_label>_v_<version>.h5`` (with ``.`` replaced by ``_``) and written to
                the file-level attribute ``dataset_version``.
                """),
            Key('dataset.labeled', BOOL, required_by=('06',),
                description="""
                When ``True``, ground-truth attributes (main halo mass, Einstein radius, velocity
                dispersion, substructure, and the deflector-only flag) are written to each
                system's group in the HDF5 file. When ``False``, they're left out of the HDF5
                file, and an answer-key CSV (``<pipeline_label>_answer_key_v_<version>.csv``) with
                each system's ``uid``, Einstein radius, substructure flag, and deflector-only flag
                is written next to it.
                """),
            Key('dataset.include_synthetic_images', BOOL, required_by=('06',),
                description="""
                When ``True``, ``_06_h5_export.py`` writes the band-matched synthetic-image arrays
                alongside each ``Exposure`` dataset.
                """),
        ),
    ),
)

KEYS = {key.name: key for section in SCHEMA for key in section.keys}

# Mappings that hold schema keys, e.g. 'survey' and 'exposure.coordinates'
_CONTAINERS = {name.rsplit('.', i)[0] for name in KEYS for i in range(1, name.count('.') + 1)}


# -- Using the schema

def apply_defaults(config):
    """Fill in the default of every absent key whose mapping exists, in place.

    Only absent keys are filled, like ``dict.get``: an explicit ``null`` stays ``null``.

    Parameters
    ----------
    config : dict
        The config, as loaded from YAML.

    Returns
    -------
    dict
        The same dict.

    Raises
    ------
    ConfigError
        If the YAML didn't load as a mapping, e.g. because the file is empty.
    """
    if not isinstance(config, dict):
        raise ConfigError(f'A config must be a YAML mapping, got {_show(config)}.')
    for key in KEYS.values():
        if not key.has_default:
            continue
        parent, _, name = key.name.rpartition('.')
        mapping = _lookup(config, parent) if parent else config
        if isinstance(mapping, dict):
            mapping.setdefault(name, key.default)
    return config


def _show(value):
    return 'null' if value is None else repr(value)


def _plain(text):
    """A condition's reStructuredText as it should read in an error message."""
    return text.replace('``', '`')


def _step_label(step):
    return '`Pipeline`' if step == 'pipeline' else f'step {step}'


def _needs(key_or_section, step, context):
    """The first requirement of ``step`` that applies, as a label, or None if none does."""
    for need in key_or_section.required_by:
        if need == step or (need == EVERY_STEP and step not in STANDALONE):
            return _step_label(step)
        if isinstance(need, Need) and need.step == step and need.when.applies(context):
            return f'{_step_label(step)} when {_plain(need.when.text)}'
    return None


def _unrecognized(config):
    """Problems with keys that aren't in the schema, and with sections that aren't mappings."""
    problems = []

    def walk(mapping, prefix):
        for name, value in mapping.items():
            dotted = f'{prefix}{name}'
            if dotted in KEYS:
                continue
            if dotted not in _CONTAINERS:
                problems.append(f'unrecognized key `{dotted}`')
            elif isinstance(value, dict):
                walk(value, f'{dotted}.')
            else:
                problems.append(f'`{dotted}`: expected a mapping, got {_show(value)}')

    walk(config, '')
    return problems


def _bad_values(config):
    """Problems with the type or value of the keys that are present, whichever step runs."""
    problems = []
    for key in KEYS.values():
        value = _lookup(config, key.name)
        if value is _MISSING:
            continue
        if key.runtime:
            problems.append(f'`{key.name}` is set by the pipeline as it runs, so remove it from the config')
        elif value is None:
            if not key.nullable:
                problems.append(f'`{key.name}`: expected {key.type.label}, got null')
        elif not key.type.check(value):
            problems.append(f'`{key.name}`: expected {key.type.label}, got {_show(value)}')
        elif key.choices and value not in key.choices:
            allowed = ' or '.join(repr(choice) for choice in key.choices)
            problems.append(f'`{key.name}`: must be {allowed}, got {_show(value)}')
    return problems


def _missing(config, steps, level):
    """Problems with required keys that are absent, one per key however many steps need it."""
    needed_by = {}
    for step, prev_step in steps:
        context = Context(config, prev_step, level)
        for section in SCHEMA:
            if section.name and _lookup(config, section.name) is _MISSING:
                label = _needs(section, step, context)
                if label:
                    needed_by.setdefault(section.name, []).append(label)
            for key in section.keys:
                if _lookup(config, key.name) is _MISSING:
                    label = _needs(key, step, context)
                    if label:
                        needed_by.setdefault(key.name, []).append(label)
    return [f'missing `{name}`, required by {", ".join(labels)}' for name, labels in needed_by.items()]


def validate_steps(config, steps, level=None, source='config'):
    """Check a config for a series of steps, reporting every problem at once.

    A null ``data_dir`` passes here, since the ``--data_dir`` argument can supply it; see
    :meth:`mejiro.pipeline.pipeline_helper.PipelineHelper.resolve_data_dir`.

    Parameters
    ----------
    config : dict
        The config, with defaults applied.
    steps : list of tuple
        ``(step, prev_step)`` for each step that will run. ``step`` is a script's
        ``SCRIPT_NAME``, and a name that isn't in :data:`STEPS` (a project script's, say) only
        has to have the keys every step needs. ``prev_step`` is the step it reads from.
    level : str, optional
        ``_05_romanisim``'s ``--level``.
    source : str, optional
        What to call the config in the error message; usually its path.

    Raises
    ------
    ConfigError
        If a required key is missing, a value has the wrong type or isn't an allowed value, or a
        key isn't recognized.
    """
    problems = _missing(config, steps, level) + _bad_values(config) + _unrecognized(config)
    if problems:
        labels = ', '.join(_step_label(step) for step, _ in steps)
        listing = '\n'.join(f'  - {problem}' for problem in problems)
        raise ConfigError(
            f'Invalid config {source!r} for {labels}:\n{listing}\n'
            f'If this config was written for v3.0.0, see "Upgrading from v3.0.0" in {UPGRADE_NOTES}.'
        )

    if (any(step == '02' for step, _ in steps) and _lookup(config, 'survey.use_real_sources') is True
            and not _lookup(config, 'survey.remap_bands')):
        logger.warning('survey.use_real_sources is True but survey.remap_bands is not set: step 02 '
                       'will stop at the first lens with catalog source images.')


def validate(config, step, prev_step=None, level=None, source='config'):
    """Check a config for one step. See :func:`validate_steps`."""
    validate_steps(config, [(step, prev_step)], level=level, source=source)
