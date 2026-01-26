import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from sys import exit
import time
import yaml
from argparse import Namespace
from pprint import pprint
from tqdm import tqdm
from multiprocessing import Pool
import warnings
from tqdm.auto import tqdm as tqdm_auto

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import healpy as hp
from corner import corner
from astropy.coordinates import SkyCoord
import astropy.units as u
from speclite import filters
import pickle
from scipy.stats import poisson

from mejiro.style import set_aas_style
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper
from mejiro.instruments.roman import Roman
from mejiro.engines.galsim_engine import GalSimEngine
from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage

set_aas_style()

# read configuration file
import mejiro
config_file = 'nancy.yaml'
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

if config['dev']:
    config['pipeline_label'] += '_dev'

args = Namespace(config='nancy.yaml')
pipeline = PipelineHelper(args, '04', 'nancy', ['roman'])
pipeline.input_dir = '/nfsdata1/bwedig/mejiro/nancy/02'
input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
print(f'Found {len(input_pickles)} input pickle(s) in {pipeline.input_dir}.')

dev = False
runs = 5
snr_threshold = 20
num_cores = 32
survey_area = 0.5
total_area = survey_area * pipeline.runs

# HEALPix uses nside parameter where npix = 12 * nside^2
if dev:
    nside = 5
else:
    nside = 25
npix = hp.nside2npix(nside)
print(f"Number of pixels: {npix}")

# HEALPix by default uses Galactic coordinates
# theta = colatitude from north galactic pole, phi = galactic longitude
theta, phi = hp.pix2ang(nside, np.arange(npix))

# Convert to Galactic l, b
l = np.rad2deg(phi)           # Galactic longitude [0, 360]
b = 90 - np.rad2deg(theta)    # Galactic latitude [-90, 90]

# Create SkyCoord in Galactic frame
coords_gal = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')

# Create a map (e.g., with pixel indices as values)
sky_map = np.arange(npix)

# Plot the map
hp.mollview(sky_map, title=f"HEALPix Sky Map (nside={nside}, npix={npix})", cmap='viridis')
hp.graticule()

# Overplot the pixel centers
# hp.projscatter(theta, phi, marker='o', color='red', s=20)

plt.savefig('figures/healpix_sky_map.png', dpi=300)
plt.close()

# Get the area of each pixel in square degrees
pixel_area = hp.nside2pixarea(nside, degrees=True)
print(f"Area per pixel: {pixel_area:.2f} sq deg")
print(f"Total sky area: {npix * pixel_area:.2f} sq deg")


if dev:
    bkg_pkl_file = 'bkg_dev.pkl'
else:
    bkg_pkl_file = 'bkg.pkl'

with open(bkg_pkl_file, 'rb') as f:
    bkg_dict = pickle.load(f)


bkg_array = np.array([bkg_dict[i][111] for i in range(npix)])


hp.mollview(np.log10(bkg_array), 
            title=r"Total Background at 1.46 $\mu$m", 
            cmap='viridis', coord='G', unit=r'$\log_{10}$(MJy/Sr)')
hp.graticule()
plt.savefig('figures/total_bkg.png', dpi=300)
plt.close()


def convert_to_cps(wavelength_microns, flux_mjy_per_sr, verbose=False):
  # Roman parameters
  pixel_scale_arcsec = 0.11  # arcsec/pixel
  collecting_area_m2 = 3.60767  # m²
  collecting_area_cm2 = collecting_area_m2 * 1e4  # cm²

  # Load Roman F146 filter
  filter_name = 'roman-F146'
  roman_filters = Roman.load_speclite_filters()
  filter_response = roman_filters[4]

  # Step 1: Convert pixel scale to solid angle (sr/pixel)
  arcsec_per_radian = 206265.0
  pixel_solid_angle_sr = (pixel_scale_arcsec / arcsec_per_radian)**2

  if verbose: 
    print(f"Pixel solid angle: {pixel_solid_angle_sr:.6e} sr/pixel")
    print(f"Pixel area: {pixel_scale_arcsec**2:.6f} arcsec²/pixel")

  # Step 2: Convert surface brightness to flux per pixel
  flux_mjy_per_pixel = flux_mjy_per_sr * pixel_solid_angle_sr

  if verbose: 
    print(f"\nExample flux at {wavelength_microns[10]:.2f} μm:")
    print(f"  Surface brightness: {flux_mjy_per_sr[10]:.4e} MJy/sr")
    print(f"  Flux per pixel: {flux_mjy_per_pixel[10]:.4e} MJy/pixel")

  # Step 3: Convert to wavelength in Angstroms
  wavelength_angstrom = wavelength_microns * 1e4  # 1 μm = 10^4 Å

  # Step 4: Convert F_ν (MJy/pixel) to f_λ (erg/s/cm²/Å/pixel)
  # Key conversions:
  #   1 Jy = 1e-23 erg/s/cm²/Hz
  #   1 MJy = 1e6 Jy = 1e-17 erg/s/cm²/Hz
  #   f_λ = f_ν × c/λ² where c is in Å/s
  c_angstrom_per_s = 2.99792458e18  # speed of light in Å/s

  # F_ν in erg/s/cm²/Hz per pixel
  flux_nu_erg = flux_mjy_per_pixel * 1e-17  # erg/s/cm²/Hz

  # Convert to f_λ in erg/s/cm²/Å
  # f_λ dλ = f_ν dν, and ν = c/λ, so dν = -c/λ² dλ
  # Therefore f_λ = f_ν × c/λ²
  flux_lambda_erg = flux_nu_erg * c_angstrom_per_s / wavelength_angstrom**2  # erg/s/cm²/Å

  if verbose: 
    print(f"\nAt {wavelength_microns[10]:.2f} μm:")
    print(f"  F_ν: {flux_nu_erg[10]:.4e} erg/s/cm²/Hz")
    print(f"  f_λ: {flux_lambda_erg[10]:.4e} erg/s/cm²/Å")

  # Step 5: Convert energy flux to photon flux
  # Photon energy E = hc/λ, so photon flux = energy flux / E = (f_λ × λ)/(hc)
  h = 6.62607015e-27  # Planck constant in erg·s
  c_cm_per_s = 2.99792458e10  # cm/s

  # Photon flux in photons/s/cm²/Å
  photon_flux = (flux_lambda_erg * wavelength_angstrom) / (h * c_angstrom_per_s)

  if verbose: print(f"  Photon flux: {photon_flux[10]:.4e} photons/s/cm²/Å")

  # Step 6: Interpolate onto filter wavelength grid
  filter_wavelength = filter_response.wavelength  # Angstroms
  filter_transmission = filter_response.response

  photon_flux_interp = np.interp(filter_wavelength, wavelength_angstrom, 
                                  photon_flux, left=0, right=0)

  # Step 7: Integrate photon flux over filter bandpass
  # counts/s/cm² = ∫ photon_flux(λ) × T(λ) dλ
  # Integration over wavelength in Angstroms gives photons/s/cm²
  counts_per_sec_per_cm2 = np.trapz(photon_flux_interp * filter_transmission, 
                                      filter_wavelength)

  # Step 8: Multiply by collecting area
  total_counts_per_sec = counts_per_sec_per_cm2 * collecting_area_cm2

  if verbose:
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS for {filter_name}:")
    print(f"{'='*60}")
    print(f"Counts/sec/cm²: {counts_per_sec_per_cm2:.6e}")
    print(f"Total counts/sec (Roman, {collecting_area_m2:.2f} m²): {total_counts_per_sec:.2f}")
    print(f"{'='*60}")
  
  return total_counts_per_sec


wavelength_microns = bkg_dict['wave_array']  # microns

bkg_file = 'bkg_cps.npy'
if dev:
    bkg_file = 'bkg_cps_dev.npy'

if os.path.exists(bkg_file):
    cps_array = np.load(bkg_file)
else:
    cps_array = []
    for i in tqdm(range(npix)):
        flux_mjy_per_sr = bkg_dict[i]
        cps = convert_to_cps(wavelength_microns, flux_mjy_per_sr, verbose=False)
        cps_array.append(cps)
    
    np.save(bkg_file, np.array(cps_array))


hp.mollview(cps_array, 
            title=r"Roman F146 Total Background from Roman Background Tool", 
            cmap='viridis', coord='G', unit=r'Counts/sec')
hp.graticule()
plt.savefig('figures/total_bkg_roman_f146.png', dpi=300)
plt.close()


roman = Roman()
engine_params = GalSimEngine.defaults(roman.name)
kwargs_numerics = SyntheticImage.DEFAULT_KWARGS_NUMERICS
if dev:
    kwargs_numerics['supersampling_factor'] = 1
else:
    kwargs_numerics['supersampling_factor'] = 5
kwargs_numerics['compute_mode'] = 'adaptive'

min_zodi_det_per_sq_deg = len(input_pickles) / total_area
print(f'Expected detectable strong lenses per sq. deg. at minimum zodiacal light: {min_zodi_det_per_sq_deg:.2f}')

expected_num_per_pixel = min_zodi_det_per_sq_deg * pixel_area
# if dev:
#     expected_num_per_pixel /= 100
rv = poisson(expected_num_per_pixel)
print(f'Expected number per healpix pixel: {expected_num_per_pixel:.4f}')

def process_cps(args):
    """Process one cps value - this runs in parallel"""
    idx, cps = args

    # Determine if this is a spot-check iteration (10 evenly spaced)
    spot_check = idx % 750 == 0  # 7500/10 = 750

    detectable_trials = []
    for run in range(runs):
        num_detectable = 0

        # randomly draw the appropriate number of strong lenses for this healpix pixel
        np.random.seed()  # ensure different seed for each iteration of runs
        num_lenses = rv.rvs()
        pkls_for_pixel = np.random.choice(input_pickles, size=num_lenses, replace=True)

        exposures_and_snrs = []

        for pkl in pkls_for_pixel:  # , disable=not dev
            # Make a local copy to avoid any potential issues with shared state
            engine_params_local = engine_params.copy()
            engine_params_local['background_cps'] = cps
            kwargs_numerics_local = kwargs_numerics.copy()

            # unpickle the lens
            lens = util.unpickle(pkl)

            # create synthetic image
            img = SyntheticImage(lens,
                                 instrument=roman,
                                 band='F146',
                                 fov_arcsec=np.max([lens.get_einstein_radius() * 4, 5]),
                                #  instrument_params=instrument_params,
                                 kwargs_numerics=kwargs_numerics_local,
                                #  kwargs_psf=kwargs_psf,
                                 pieces=True,
                                 verbose=False)

            exposure = Exposure(img,
                        exposure_time=30,
                        engine='galsim',
                        engine_params=engine_params_local,
                        verbose=False)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                snr = exposure.get_snr()

            if snr >= snr_threshold:
                num_detectable += 1

            exposures_and_snrs.append((exposure, snr))
        
        # Save figure for spot-checking
        if spot_check and run == 0:  # Only first run to avoid too many figures
            fig, axes = plt.subplots(7, 7, figsize=(20, 20))
            for ax, (exposure, snr) in zip(axes.flatten(), exposures_and_snrs):
                ax.imshow(exposure.exposure, norm=LogNorm(), cmap='cubehelix')
                ax.set_title(f'SNR: {snr:.2f}')
                ax.axis('off')
            plt.suptitle(f'healpix pixel {idx}, background level {cps:.2f} counts/sec/pixel, detectable lenses: {num_detectable} of {num_lenses} ({(num_detectable/num_lenses*100) if num_lenses > 0 else 0:.2f}%)', fontsize=16)
            plt.savefig(f'figures/spot_check_{idx:04d}.png', dpi=300)
            plt.close()
            spot_check = False  # Only one figure per pixel

        # calculate detectable fraction
        detectable_per_sq_deg = num_detectable / pixel_area
        detectable_trials.append(detectable_per_sq_deg)

    mean_detectable = np.mean(detectable_trials)
    return mean_detectable

def init_worker():
    """
    Initialize each worker with a unique random seed based on process ID and time.

    When using multiprocessing.Pool with the default fork start method (on Linux/Mac), each worker process inherits a copy of the parent process's memory state, including NumPy's random number generator state.
    """
    seed = int(time.time() * 1000) + os.getpid()
    np.random.seed(seed % (2**32))  # NumPy seed must be < 2^32

# Parallelize the outer loop with order preservation
with Pool(num_cores, initializer=init_worker) as pool:
    detectable_counts = list(tqdm_auto(
        pool.imap(process_cps, enumerate(cps_array)),
        total=len(cps_array),
        desc='Background levels'
    ))

hp.mollview(np.array(detectable_counts), 
            title=f"Detectable Strong Lenses per sq. deg. (SNR $\geq$ {snr_threshold})", 
            cmap='viridis', coord='G', unit=r'Detectable Lenses per deg$^2$')
hp.graticule()
plt.savefig('figures/nancy_detectable_sky_map.png', dpi=300)
plt.close()

if dev:
    np.save('nancy_detectable_counts_dev.npy', np.array(detectable_counts))
else:
    np.save('nancy_detectable_counts.npy', np.array(detectable_counts))
