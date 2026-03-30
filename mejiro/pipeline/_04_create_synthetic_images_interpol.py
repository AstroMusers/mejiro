"""
Generates synthetic images using a pre-computed interpolated deflection map.

This is an optimized variant of _04_create_synthetic_images.py. Instead of ray-tracing
independently for each photometric band, it evaluates the lens model once to build an
interpolated deflection map (using lenstronomy's INTERPOL lens profile), then reuses that
map for all bands. The deflection field depends only on the mass model and is
band-independent, so this avoids redundant evaluations of complex lens profiles (NFW,
Hernquist, subhalos, etc.).

Usage:
    python3 _04_create_synthetic_images_interpol.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Util import util as lenstronomy_util
from tqdm import tqdm

from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '04'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # retrieve configuration parameters
    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']

    # set up jaxstronomy
    if pipeline.config['jaxtronomy']['use_jax']:
        os.environ['JAX_PLATFORM_NAME'] = pipeline.config['jaxtronomy'].get('jax_platform', 'cpu')

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo' or pipeline.instrument_name == 'jwst':
        input_pickles = pipeline.retrieve_pickles(prefix='lens', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    logger.info(f'Processing {count} lens(es)')

    # tuple the parameters
    tuple_list = [(pipeline, synthetic_image_config, psf_config, input_pickle) for input_pickle in input_pickles]

    # submit tasks to the executor
    with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
        futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def _compute_interpol_deflection_map(lens, fov_arcsec, pixel_scale, supersampling_factor):
    """Compute the deflection map on a fine grid and return INTERPOL kwargs.

    Evaluates the full lens model (alpha, potential, hessian) on a supersampled
    grid and packages the results as a dictionary suitable for lenstronomy's
    INTERPOL lens profile.

    Parameters
    ----------
    lens : StrongLens
        The strong lens system with analytic lens model.
    fov_arcsec : float
        Field of view in arcseconds.
    pixel_scale : float
        Pixel scale in arcsec/pix (use the finest scale across bands).
    supersampling_factor : int
        Supersampling factor for the interpolation grid resolution.

    Returns
    -------
    dict
        INTERPOL kwargs dictionary with keys: grid_interp_x, grid_interp_y,
        f_, f_x, f_y, f_xx, f_yy, f_xy.
    """
    num_pix_grid = util.set_odd_num_pix(fov_arcsec, pixel_scale)
    adjusted_fov = num_pix_grid * pixel_scale
    num_pix_interp = num_pix_grid * supersampling_factor

    interp_coords = np.linspace(-adjusted_fov / 2, adjusted_fov / 2, num_pix_interp)
    xx, yy = np.meshgrid(interp_coords, interp_coords)
    x_flat, y_flat = xx.ravel(), yy.ravel()

    lens_model = lens.lens_model
    kwargs_lens = lens.kwargs_lens

    alpha_x, alpha_y = lens_model.alpha(x_flat, y_flat, kwargs_lens)
    potential = lens_model.potential(x_flat, y_flat, kwargs_lens)
    f_xx, f_xy, _, f_yy = lens_model.hessian(x_flat, y_flat, kwargs_lens)

    shape = (num_pix_interp, num_pix_interp)
    return {
        'grid_interp_x': interp_coords,
        'grid_interp_y': interp_coords,
        'f_': potential.reshape(shape),
        'f_x': alpha_x.reshape(shape),
        'f_y': alpha_y.reshape(shape),
        'f_xx': f_xx.reshape(shape),
        'f_yy': f_yy.reshape(shape),
        'f_xy': f_xy.reshape(shape),
    }


def _precompute_adaptive_grid(lens, band, instrument, fov_arcsec, pad=40):
    """Pre-compute the adaptive supersampling grid for a given band.

    This replicates the logic in SyntheticImage.build_adaptive_grid() but
    uses the original (non-INTERPOL) lens model, which is necessary because
    INTERPOL kwargs lack center_x/center_y.

    Parameters
    ----------
    lens : StrongLens
        The strong lens system with analytic lens model.
    band : str
        Photometric band.
    instrument : Instrument
        Instrument object.
    fov_arcsec : float
        Field of view in arcseconds.
    pad : int
        Padding for the adaptive grid mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask for supersampled pixels.
    """
    pixel_scale = instrument.get_pixel_scale(band).value
    num_pix = util.set_odd_num_pix(fov_arcsec, pixel_scale)

    # set up coordinate transform for this band
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
        lenstronomy_util.make_grid_with_coordtransform(
            numPix=num_pix, deltapix=pixel_scale, subgrid_res=1,
            left_lower=False, inverse=False))
    coords = Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)

    # get image positions in pixel coordinates
    image_x, image_y = lens.get_image_positions()
    img_pix_x, img_pix_y = coords.map_coord2pix(ra=image_x, dec=image_y)

    # compute radial distances of images from center
    image_radii = []
    for x, y in zip(img_pix_x, img_pix_y):
        image_radii.append(np.sqrt((x - (num_pix // 2)) ** 2 + (y - (num_pix // 2)) ** 2))

    # build distance grid
    x = np.linspace(-num_pix // 2, num_pix // 2, num_pix)
    y = np.linspace(-num_pix // 2, num_pix // 2, num_pix)
    X, Y = np.meshgrid(x, y)
    center_x = lens.kwargs_lens[0].get('center_x', 0)
    center_y = lens.kwargs_lens[0].get('center_y', 0)
    distance = np.sqrt((X - (center_x / pixel_scale)) ** 2 + (Y - (center_y / pixel_scale)) ** 2)

    min_r = np.min(image_radii) - pad
    if min_r < 0:
        min_r = 0
    max_r = np.max(image_radii) + pad
    if max_r > num_pix // 2:
        max_r = num_pix // 2

    return (distance >= min_r) & (distance <= max_r)


def create_synthetic_image(input):
    # unpack tuple
    (pipeline, synthetic_image_config, psf_config, input_pickle) = input

    # unpack pipeline params
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    pieces = synthetic_image_config['pieces']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')  # Roman-specific parameter, not required for HWO

    # unpickle the lens
    lens = util.unpickle(input_pickle)

    # build kwargs_numerics
    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode
    }

    get_psf_args = {}
    instrument_params = {}

    # set detector and pick random position
    if pipeline.instrument_name == 'roman':
        possible_detector_positions = roman_util.divide_up_sca(divide_up_detector)
        detector_position = random.choice(possible_detector_positions)
        instrument_params['detector'] = pipeline.parse_sca_from_filename(input_pickle)
        instrument_params['detector_position'] = detector_position
        get_psf_args |= instrument_params

        sca_string = roman_util.get_sca_string(instrument_params['detector']).lower()
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    # --- Compute interpolated deflection map once ---
    pixel_scales = {b: pipeline.instrument.get_pixel_scale(b).value for b in bands}
    finest_pixel_scale = min(pixel_scales.values())

    logger.info(f'Computing interpolated deflection map for lens {lens.name}')
    interpol_kwargs = _compute_interpol_deflection_map(
        lens, fov_arcsec, finest_pixel_scale, supersampling_factor
    )

    # store on lens for persistence
    lens.interpol_deflection_map = interpol_kwargs

    # --- Pre-compute adaptive grids if needed ---
    adaptive_grids = {}
    if supersampling_compute_mode == 'adaptive':
        for band in bands:
            adaptive_grids[band] = _precompute_adaptive_grid(
                lens, band, pipeline.instrument, fov_arcsec
            )

    # --- Swap lens model to INTERPOL ---
    original_lens_model_list = lens.lens_model_list
    original_kwargs_lens = lens.kwargs_lens
    original_use_jax = lens.use_jax

    lens.lens_model_list = ['INTERPOL']
    lens.kwargs_lens = [interpol_kwargs]
    lens.use_jax = [False]

    # --- Generate synthetic images for each band ---
    for band in bands:
        # get PSF
        get_psf_args |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        # include pre-computed adaptive grid if applicable
        band_kwargs_numerics = kwargs_numerics.copy()
        if supersampling_compute_mode == 'adaptive' and band in adaptive_grids:
            band_kwargs_numerics['supersampled_indexes'] = adaptive_grids[band]

        # build and pickle synthetic image
        try:
            synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=pipeline.instrument,
                                        band=band,
                                        fov_arcsec=fov_arcsec,
                                        instrument_params=instrument_params,
                                        kwargs_numerics=band_kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=pieces)
            util.pickle(os.path.join(output_dir, f'SyntheticImage_{lens.name}_{band}.pkl'), synthetic_image)
        except Exception as e:
            failed_pickle_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. Pickling to {failed_pickle_path}')
            return

    # --- Restore original lens model ---
    lens.lens_model_list = original_lens_model_list
    lens.kwargs_lens = original_kwargs_lens
    lens.use_jax = original_use_jax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic images with interpolated deflection map")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
