import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import matplotlib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from hydra import initialize, compose
import pickle
from glob import glob
from pprint import pprint
from tqdm import tqdm
import galsim
from copy import deepcopy
import random
import json
from lenstronomy.Util import correlation
from pyHalo.preset_models import CDM
from matplotlib.lines import Line2D
from lenstronomy.Plots import plot_util


def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


def plot_autocorrelation(x):
    return np.arange(len(x)), autocorrelation(x)


def plot_deflection_field(ax, lens, size=20, fov=5, cmap='viridis'):
    x = np.linspace(-fov / 2, fov / 2, size)
    y = np.linspace(-fov / 2, fov / 2, size)
    xx, yy = np.meshgrid(x, y)

    potential = lens.lens_model_class.potential(xx.flatten(), yy.flatten(), lens.kwargs_lens)

    # Compute the gradient of the potential (deflection field)
    potential_reshaped = potential.reshape((size, size))
    grad_y, grad_x = np.gradient(potential_reshaped)

    im = ax.imshow(potential_reshaped, extent=[-fov / 2, fov / 2, -fov / 2, fov / 2], cmap=cmap)
    ax.quiver(xx, yy, grad_x, grad_y, color='white')
    return im


def plot_residual_deflection_field(ax, lens_with, lens_without, size=20, fov=5, cmap='viridis', arrow_color='white'):
    x = np.linspace(-fov / 2, fov / 2, size)
    y = np.linspace(-fov / 2, fov / 2, size)
    xx, yy = np.meshgrid(x, y)

    potential_with = lens_with.lens_model_class.potential(xx.flatten(), yy.flatten(), lens_with.kwargs_lens)
    potential_without = lens_without.lens_model_class.potential(xx.flatten(), yy.flatten(), lens_without.kwargs_lens)
    potential_residual = potential_with - potential_without

    # Compute the gradient of the potential (deflection field)
    potential_reshaped = potential_residual.reshape((size, size))
    grad_y, grad_x = np.gradient(potential_reshaped)

    im = ax.imshow(potential_reshaped, extent=[-fov / 2, fov / 2, -fov / 2, fov / 2], cmap=cmap)
    ax.quiver(xx, yy, grad_x, grad_y, color=arrow_color)
    return im


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # os.nice(19)

    # Enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    import mejiro
    from mejiro.plots import overplot
    from mejiro.utils import util
    from mejiro.lenses import lens_util
    from mejiro.utils import util
    from mejiro.engines import webbpsf_engine
    from mejiro.instruments.roman import Roman

    # set matplotlib style
    plt.style.use(f'{config.machine.repo_dir}/mejiro/mplstyle/science.mplstyle')

    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    # debugging = True  # TODO TEMP

    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    detectable_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, verbose=False)

    num_figures = 30
    band = 'F129'
    snr_cut = 100
    logm_cut = 13
    sca = 1
    sca_position = (2048, 2048)
    instrument_params = {
        'detector': sca,
        'detector_position': sca_position
    }
    roman = Roman()
    rng = galsim.UniformDeviate(42)
    num_pix = 72  # pipeline_params['num_pix']
    side = 7.92  # pipeline_params['side']
    grid_oversample = 5 # pipeline_params['grid_oversample']
    final_pixel_side = 66  # pipeline_params['final_pixel_side']
    exposure_time = 584  # pipeline_params['exposure_time']
    supersampling_factor = pipeline_params['supersampling_factor']
    supersampling_compute_mode = pipeline_params['supersampling_compute_mode']

    # get PSF
    psf_cache_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    psf_id = webbpsf_engine.get_psf_id(band, sca, sca_position, grid_oversample, 101)
    psf = webbpsf_engine.get_cached_psf(psf_id, psf_cache_dir, verbose=False)

    # get lenses
    lenses = [l for l in detectable_lenses if l.snr > snr_cut and np.log10(l.main_halo_mass) > logm_cut]  #  and l.lensed_source_mags['F129'] < 20

    arg_list = [(lens, psf, roman, band, side, grid_oversample, instrument_params, exposure_time, num_pix) for lens in lenses[:num_figures]]

    # Determine the number of processes
    count = len(arg_list)
    cpu_count = multiprocessing.cpu_count()
    process_limit = cpu_count // 2
    if num_figures < process_limit:
        process_count = num_figures
    else:
        process_count = process_limit
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s) to generate {count} figure(s)')

    # Process the tasks with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(generate_figures, args) for args in arg_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    util.print_execution_time(start, stop)


def generate_figures(args):
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure
    from mejiro.lenses import lens_util
    from mejiro.plots import overplot

    # Unpack tuple
    lens, psf, roman, band, side, grid_oversample, instrument_params, exposure_time, num_pix = args

    # lens with smooth mass model
    lens_without = deepcopy(lens)
    synth_without = SyntheticImage(lens_without,
                                    roman,
                                    band=band,
                                    arcsec=side,
                                    oversample=grid_oversample,
                                    instrument_params=instrument_params,
                                    pieces=False,
                                    verbose=False)
    engine_params = {
        'detector_effects': False,
        'sky_background': False
    }
    exposure_without = Exposure(synth_without,
                                    exposure_time=exposure_time,
                                    engine_params=engine_params,
                                    psf=psf,
                                    verbose=False)
    
    # generate subhalo realizations
    realization_6_8 = CDM(round(lens.z_lens, 2),
                    round(lens.z_source, 2),
                    log_mlow=6,
                    log_mhigh=8,
                    log_m_host=np.log10(lens.main_halo_mass),
                    cone_opening_angle_arcsec=5. * np.sqrt(2),
                    LOS_normalization=0.)
    realization_8_9 = CDM(round(lens.z_lens, 2),
                    round(lens.z_source, 2),
                    log_mlow=8,
                    log_mhigh=9,
                    log_m_host=np.log10(lens.main_halo_mass),
                    cone_opening_angle_arcsec=5. * np.sqrt(2),
                    LOS_normalization=0.)
    realization_9_10 = CDM(round(lens.z_lens, 2),
                    round(lens.z_source, 2),
                    log_mlow=9,
                    log_mhigh=10,
                    log_m_host=np.log10(lens.main_halo_mass),
                    cone_opening_angle_arcsec=5. * np.sqrt(2),
                    LOS_normalization=0.)
    wdm = deepcopy(realization_9_10)
    mdm = wdm.join(realization_8_9)
    cdm = mdm.join(realization_6_8)

    # lens with CDM subhalos
    lens_with = deepcopy(lens)
    lens_with.add_subhalos(cdm, mass_sheet_correction=False)
    synth_with = SyntheticImage(lens_with,
                                roman,
                                band=band,
                                arcsec=side,
                                oversample=grid_oversample,
                                instrument_params=instrument_params,
                                verbose=False)
    exposure_with = Exposure(synth_with,
                            exposure_time=exposure_time,
                            engine_params=engine_params,
                            psf=psf,
                            verbose=False)
    
    # lens with MDM subhalos
    lens_mdm = deepcopy(lens)
    lens_mdm.add_subhalos(mdm, mass_sheet_correction=False)
    synth_mdm = SyntheticImage(lens_mdm,
                                roman,
                                band=band,
                                arcsec=side,
                                oversample=grid_oversample,
                                instrument_params=instrument_params,
                                verbose=False)
    exposure_mdm = Exposure(synth_mdm,
                            exposure_time=exposure_time,
                            engine_params=engine_params,
                            psf=psf,
                            verbose=False)
    
    # lens with WDM subhalos
    lens_wdm = deepcopy(lens)
    lens_wdm.add_subhalos(wdm, mass_sheet_correction=False)
    synth_wdm = SyntheticImage(lens_wdm,
                                roman,
                                band=band,
                                arcsec=side,
                                oversample=grid_oversample,
                                instrument_params=instrument_params,
                                verbose=False)
    exposure_wdm = Exposure(synth_wdm,
                            exposure_time=exposure_time,
                            engine_params=engine_params,
                            psf=psf,
                            verbose=False)
    
    wdm_image = exposure_wdm.exposure  # synth_wdm.image
    mdm_image = exposure_mdm.exposure  # synth_mdm.image
    cdm_image = exposure_with.exposure  # synth_with.image
    smooth_image = exposure_without.exposure  # synth_without.image

    # calculate power spectra of residuals
    ps1d_residual_wdm, r = correlation.power_spectrum_1d(wdm_image - smooth_image)
    ps1d_residual_mdm, _ = correlation.power_spectrum_1d(mdm_image - smooth_image)
    ps1d_residual_cdm, _ = correlation.power_spectrum_1d(cdm_image - smooth_image)
    ps2d_residual = correlation.power_spectrum_2d(cdm_image - smooth_image)
    ps2d_cdm = correlation.power_spectrum_2d(cdm_image)
    ps2d_smooth = correlation.power_spectrum_2d(smooth_image)

    # calculate subhalo mass fractions
    f_wdm = wdm.masses.sum() / lens.main_halo_mass
    f_mdm = mdm.masses.sum() / lens.main_halo_mass
    f_cdm = cdm.masses.sum() / lens.main_halo_mass
    
    # plot
    f, ax = plt.subplots(3, 3, figsize=(12, 9))  # , constrained_layout=True

    num_subhalos = 10
    coords = lens_util.get_coords(num_pix)
    cdm_subhalos = cdm.halos
    cdm_subhalos_sorted = sorted(cdm_subhalos, key=lambda halo: halo.mass, reverse=True)
    # print([f'{h.mass:.2e}' for h in cdm_subhalos_sorted[num_subhalos]])

    im = ax[0, 0].imshow(np.log10(smooth_image), cmap='viridis')
    overplot.source_position(ax[0, 0], lens_without, coords, alpha=1, color='orange', size=7.5)
    overplot.lens_position(ax[0, 0], lens_without, coords, alpha=1, color='red', size=7.5)
    overplot.caustics(ax[0, 0], lens_without, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='g')
    overplot.critical_curves(ax[0, 0], lens_without, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='b')
    plot_util.scale_bar(ax[0, 0], d=smooth_image.shape[0], dist=1 / 0.11, text='1"', color='w', flipped=True)
    ax[0, 0].legend()
    ax[0, 0].set_title('Count rate with only a smooth main halo')
    ax[0, 0].axis('off')
    plt.colorbar(im, ax=ax[0, 0], label=r'$\log_{10}$(Counts/Sec)')

    im = ax[0, 1].imshow(np.log10(cdm_image), cmap='viridis')
    overplot.source_position(ax[0, 1], lens_with, coords, alpha=1, color='orange', size=7.5)
    overplot.lens_position(ax[0, 1], lens_with, coords, alpha=1, color='red', size=7.5)
    overplot.caustics(ax[0, 1], lens_with, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='g')
    overplot.critical_curves(ax[0, 1], lens_with, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='b')
    plot_util.scale_bar(ax[0, 1], d=smooth_image.shape[0], dist=1 / 0.11, text='1"', color='w', flipped=True)
    for halo in cdm_subhalos_sorted[:num_subhalos]:
        x, y = coords.map_coord2pix(ra=halo.x, dec=halo.y)
        markersize = (halo.mass - 1e8) * (50 - 5) / (1e10 - 1e8) + 5
        ax[0, 1].plot(x, y, 'o', markersize=markersize, color='black', markerfacecolor='none')
    ax[0, 1].set_title('Count rate with additional CDM subhalos')
    ax[0, 1].axis('off')
    plt.colorbar(im, ax=ax[0, 1], label=r'$\log_{10}$(Counts/Sec)')

    im = ax[0, 2].imshow(cdm_image - smooth_image, norm=colors.CenteredNorm(), cmap='bwr')
    overplot.source_position(ax[0, 2], lens_with, coords, alpha=1, color='orange', size=7.5)
    overplot.lens_position(ax[0, 2], lens_with, coords, alpha=1, color='red', size=7.5)
    overplot.caustics(ax[0, 2], lens_with, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='g')
    overplot.critical_curves(ax[0, 2], lens_with, coords, num_pix, delta_pix=0.11, linewidth=1.5, alpha=0.6, color='b')
    for halo in cdm_subhalos_sorted[:num_subhalos]:
        x, y = coords.map_coord2pix(ra=halo.x, dec=halo.y)
        markersize = (halo.mass - 1e8) * (50 - 5) / (1e10 - 1e8) + 5
        ax[0, 2].plot(x, y, 'o', markersize=markersize, color='black', markerfacecolor='none')
    ax[0, 2].set_title('Residual count rate')
    ax[0, 2].axis('off')
    plt.colorbar(im, ax=ax[0, 2], label='Counts/Sec')

    bins = 50 # int(np.sqrt(synth_without.image.shape[0] * synth_without.image.shape[1]))
    ax[1, 0].hist((wdm_image - smooth_image).flatten(), bins=bins, histtype='step', label=r'$10^9-10^{10}$ M$_\odot$, $f_{sub}$=' + f'{f_wdm:.4f}')
    ax[1, 0].hist((mdm_image - smooth_image).flatten(), bins=bins, histtype='step', label=r'$10^8-10^{10}$ M$_\odot$, $f_{sub}$=' + f'{f_mdm:.4f}')
    ax[1, 0].hist((cdm_image - smooth_image).flatten(), bins=bins, histtype='step', label=r'$10^6-10^{10}$ M$_\odot$, $f_{sub}$=' + f'{f_cdm:.4f}')
    ax[1, 0].set_title('Histogram of the residual count rate')
    ax[1, 0].set_xlabel('Counts/Sec/Pixel')
    ax[1, 0].set_ylabel('Number of pixels')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylim(1, 1e6)
    ax[1, 0].legend()

    ax[1, 1].plot(r, ps1d_residual_wdm)
    ax[1, 1].plot(r, ps1d_residual_mdm)
    ax[1, 1].plot(r, ps1d_residual_cdm)
    ax[1, 1].set_title('1D power spectrum and autocorrelation\nof the residual count rate')
    ax[1, 1].set_xlabel(r'$k$ [Pixels$^{-1}$]')
    ax[1, 1].set_ylabel(r'$P(k)$')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    solid_line = Line2D([0], [0], color='black', linestyle='-', label='Solid Line')
    dotted_line = Line2D([0], [0], color='black', linestyle=':', label='Dotted Line')
    ax[1, 1].legend([solid_line, dotted_line], ['Power Spectrum', 'Autocorrelation'])

    twinxy = ax[1, 1].twinx()
    secax = ax[1, 1].secondary_xaxis('top')
    secax.set_xlabel(r'$\theta$ [arcsec]')
    secax.set_xticks(ax[1, 1].get_xticks())
    secax.set_xticklabels([f'{0.11 * tick:.2f}' for tick in ax[1, 1].get_xticks()])
    twinxy.set_xscale('log')
    twinxy.set_yscale('log')
    twinxy.set_ylabel('Autocorrelation')
    twinxy.plot(*plot_autocorrelation(ps1d_residual_wdm), color='C0', linestyle=':')
    twinxy.plot(*plot_autocorrelation(ps1d_residual_mdm), color='C1', linestyle=':')
    twinxy.plot(*plot_autocorrelation(ps1d_residual_cdm), color='C2', linestyle=':')

    im = ax[1, 2].imshow(np.log10(ps2d_residual), cmap='inferno')
    ax[1, 2].set_title('2D power spectrum of residual count rate')
    ax[1, 2].set_xlabel(r'$k_x$ [Pixels$^{-1}$]')
    ax[1, 2].set_ylabel(r'$k_y$ [Pixels$^{-1}$]')
    ax[1, 2].set_aspect('equal')
    plt.colorbar(im, ax=ax[1, 2], label=r'$\log_{10}$(Power)')

    im = ax[2, 0].imshow(np.log10(ps2d_cdm - ps2d_smooth), cmap='bwr', norm=colors.CenteredNorm())
    ax[2, 0].set_title('Residual of 2D power spectra')
    ax[2, 0].set_xlabel(r'$k_x$ [Pixels$^{-1}$]')
    ax[2, 0].set_ylabel(r'$k_y$ [Pixels$^{-1}$]')
    ax[2, 0].set_aspect('equal')
    plt.colorbar(im, ax=ax[2, 0], label=r'$\log_{10}$(Power)')

    im = plot_deflection_field(ax[2, 1], lens_with, size=15, fov=side, cmap='viridis')
    ax[2, 1].set_title('Lensing potential and deflection field')
    ax[2, 1].axis('off')
    plt.colorbar(im, ax=ax[2, 1], label=r'Lensing potential [arcsec$^2$]')

    im = plot_residual_deflection_field(ax[2, 2], lens_with, lens_without, size=15, fov=side, cmap='viridis', arrow_color='white')
    ax[2, 2].set_title('Residual lensing potential\nand deflection field')
    ax[2, 2].axis('off')
    plt.colorbar(im, ax=ax[2, 2], label=r'Lensing potential [arcsec$^2$]')

    # plt.suptitle(r'$z_{lens}$=' + f'{lens.z_lens:.2f}, ' + r'$z_{source}$=' + f'{lens.z_source:.2f}, {exposure_time} s exposure with {band} filter on SCA{str(sca).zfill(2)}, 5x5 arcsec cutouts')
    plt.tight_layout()
    plt.savefig(f'/data/bwedig/mejiro/output/TD_figures/lens_{lens.uid}.png')
    plt.close()


if __name__ == '__main__':
    main()
