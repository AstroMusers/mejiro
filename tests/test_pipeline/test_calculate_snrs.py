"""calculate_snrs rebuilds SNR exposures with GalSim, whatever ``imaging.engine`` says.

``imaging.engine`` only selects the step 05 simulator. ``Exposure`` has no romanisim engine,
so passing the config value through to it crashed ``Pipeline.run()`` right after
``_05_romanisim``.
"""
import copy
import math
import os
from types import SimpleNamespace

import yaml

import mejiro
from mejiro.galaxy_galaxy import SampleGG
from mejiro.pipeline import calculate_snrs
from mejiro.utils import util

TEST_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'test.yaml')
BAND = 'F106'


def _calculate_snr(config, pipeline_dir, instrument, psf_cache_dir):
    """Run ``calculate_snr`` for one system, from the files steps 02 and 04 would leave."""
    lens_dir = pipeline_dir / '02' / 'sca01'
    synth_dir = pipeline_dir / '04' / 'sca01'
    lens_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    lens = SampleGG()
    util.pickle(str(lens_dir / f'lens_{lens.name}.pkl'), lens)
    # _rebuild_snr only reads instrument_params from the step 04 output; this position is
    # the one tests/test_data has a cached PSF for
    util.pickle(str(synth_dir / f'SyntheticImage_{lens.name}_{BAND}.pkl'),
                SimpleNamespace(instrument_params={'detector': 1, 'detector_position': (2043, 2043)}))

    pipeline = SimpleNamespace(
        config=config,
        instrument=instrument,
        instrument_name='roman',
        pipeline_dir=str(pipeline_dir),
        psf_cache_dir=psf_cache_dir,
        step_dir=lambda step: str(pipeline_dir / step),
    )
    # the exposure file is only used for its name; the SNR is rebuilt from steps 02 and 04
    exposure_path = str(pipeline_dir / '05_romanisim' / 'sca01' / f'Exposure_{lens.name}_{BAND}.npy')
    return calculate_snrs.calculate_snr((pipeline, config['snr'], exposure_path))


def test_snr_rebuild_ignores_imaging_engine(tmp_path, roman_instrument, test_data_dir):
    with open(TEST_CONFIG) as f:
        galsim_config = yaml.safe_load(f)
    romanisim_config = copy.deepcopy(galsim_config)
    romanisim_config['imaging']['engine'] = 'romanisim'

    name, snr = _calculate_snr(romanisim_config, tmp_path, roman_instrument, test_data_dir)
    assert name == f'SampleGG_{BAND}'
    assert math.isfinite(snr) and snr > 0

    # engine_params (and so the seed) is shared, so the two configs must agree exactly
    _, galsim_snr = _calculate_snr(galsim_config, tmp_path, roman_instrument, test_data_dir)
    assert snr == galsim_snr
