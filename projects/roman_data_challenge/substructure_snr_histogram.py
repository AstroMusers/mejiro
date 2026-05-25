"""
Compute a per-system "SNR of substructure signal" = sqrt(chi^2) across a
Roman pipeline run (selected via the config's pipeline_label) and produce
a histogram per band.

For each SyntheticImage (which has substructure baked in from step _03):
    1. Render an Exposure with substructure (using the YAML's engine_params).
    2. Build a no-substructure SyntheticImage by swapping the strong_lens's
       active lens-model lists/kwargs/model with the *_macromodel backups
       saved by add_realization().
    3. Render a no-substructure Exposure FIRST with the YAML's engine_params,
       capture its galsim.Image noise components, then re-render the
       with-substructure Exposure passing those same noise objects back in
       so both exposures share an identical noise realization.
    4. Build an SNR mask from the no-substructure exposure
       (95th-percentile of source_data / sqrt(data)).
    5. chi^2 = sum_i (D_with[i] - D_no[i])^2 / D_no[i] over masked pixels.
    6. snr = sqrt(chi^2).

Outputs (under <data_dir>/<pipeline_label>/analysis/):
    substructure_snr_<band>.npy
    substructure_snr_<band>_uids.npy
    substructure_snr_failures_<band>.json
    substructure_snr_hist_<band>.png
"""
import argparse
import glob
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm


logger = logging.getLogger(__name__)


def discover_synth_pickles(pipeline_dir, band):
    pattern = os.path.join(pipeline_dir, '04', 'sca[0-1][0-9]',
                           f'SyntheticImage_*_{band}.pkl')
    return sorted(glob.glob(pattern))


def uid_from_path(synth_path):
    # SyntheticImage_<pipeline_label>_<8-digit-uid>_<band>.pkl
    base = os.path.basename(synth_path)
    return base.replace('SyntheticImage_', '').rsplit('_', 1)[0]


_INSTRUMENT_FACTORY = {
    'Roman': ('mejiro.instruments.roman', 'Roman'),
    'HST': ('mejiro.instruments.hst', 'HST'),
    'LSST': ('mejiro.instruments.lsst', 'LSST'),
    'JWST': ('mejiro.instruments.jwst', 'JWST'),
    'HWO': ('mejiro.instruments.hwo', 'HWO'),
}


def get_instrument(synth):
    instrument = getattr(synth, 'instrument', None)
    if instrument is not None:
        return instrument
    name = getattr(synth, 'instrument_name', None)
    if name not in _INSTRUMENT_FACTORY:
        raise RuntimeError(f'cannot reconstruct instrument from name={name!r}')
    import importlib
    module_path, cls_name = _INSTRUMENT_FACTORY[name]
    instrument = getattr(importlib.import_module(module_path), cls_name)()
    synth.instrument = instrument
    return instrument


def build_kwargs_psf(synth, psf_cache_dir, supersampling_factor, num_pix):
    return get_instrument(synth).get_psf_kwargs(
        band=synth.band,
        detector=synth.instrument_params['detector'],
        detector_position=synth.instrument_params['detector_position'],
        oversample=supersampling_factor,
        num_pix=num_pix,
        check_cache=True,
        psf_cache_dir=psf_cache_dir,
        require_cached=True,
    )


def make_no_substructure_synth(synth_with_sub, kwargs_psf):
    from mejiro.synthetic_image import SyntheticImage

    sl = deepcopy(synth_with_sub.strong_lens)
    for attr in ('kwargs_lens_macromodel', 'lens_model_list_macromodel',
                 'lens_redshift_list_macromodel', 'lens_model_macromodel'):
        if getattr(sl, attr, None) is None:
            raise RuntimeError(f'{attr} missing on strong_lens')

    sl.kwargs_lens = deepcopy(sl.kwargs_lens_macromodel)
    sl.lens_model_list = deepcopy(sl.lens_model_list_macromodel)
    sl.lens_redshift_list = deepcopy(sl.lens_redshift_list_macromodel)
    sl.realization = None
    sl.use_jax = list(sl.use_jax)[:len(sl.lens_model_list)]

    return SyntheticImage(
        strong_lens=sl,
        instrument=get_instrument(synth_with_sub),
        band=synth_with_sub.band,
        fov_arcsec=synth_with_sub.fov_arcsec,
        instrument_params=synth_with_sub.instrument_params,
        kwargs_numerics=synth_with_sub.kwargs_numerics,
        kwargs_psf=kwargs_psf,
        pieces=True,
    )


def compute_substructure_snr(task):
    from mejiro.analysis import snr_calculation, stats
    from mejiro.exposure import Exposure
    from mejiro.utils import util

    (synth_path, exposure_time, base_engine_params,
     psf_cache_dir, supersampling_factor, num_pix, snr_quantile) = task

    uid = uid_from_path(synth_path)

    try:
        synth_with_sub = util.unpickle(synth_path)
        band = synth_with_sub.band

        kwargs_psf = build_kwargs_psf(synth_with_sub, psf_cache_dir,
                                      supersampling_factor, num_pix)
        try:
            synth_no_sub = make_no_substructure_synth(synth_with_sub, kwargs_psf)
        except RuntimeError as e:
            if 'macromodel missing on strong_lens' in str(e):
                return {'uid': uid, 'band': band, 'snr': None,
                        'error': str(e), 'skipped': True}
            raise

        exp_no_sub = Exposure(synth_no_sub,
                              exposure_time=exposure_time,
                              engine='galsim',
                              engine_params=deepcopy(base_engine_params))

        reuse_params = {
            'rng_seed': base_engine_params.get('rng_seed'),
            'min_zodi_factor': base_engine_params.get('min_zodi_factor'),
            'sky_background': base_engine_params.get('sky_background', True),
            'detector_effects': base_engine_params.get('detector_effects', True),
            'poisson_noise': exp_no_sub.poisson_noise,
            'reciprocity_failure': exp_no_sub.reciprocity_failure,
            'dark_noise': exp_no_sub.dark_noise,
            'nonlinearity': exp_no_sub.nonlinearity,
            'ipc': exp_no_sub.ipc,
            'read_noise': exp_no_sub.read_noise,
        }
        exp_with_sub = Exposure(synth_with_sub,
                                exposure_time=exposure_time,
                                engine='galsim',
                                engine_params=reuse_params)

        snr_arr = snr_calculation.get_snr_array(exp_no_sub)
        if not np.isfinite(snr_arr).any():
            return {'uid': uid, 'band': band, 'snr': None,
                    'error': 'snr array all non-finite'}
        threshold = np.quantile(snr_arr, snr_quantile)
        if not np.isfinite(threshold):
            return {'uid': uid, 'band': band, 'snr': None,
                    'error': 'snr threshold non-finite'}
        masked = np.ma.masked_where(snr_arr <= threshold, snr_arr)
        if masked.count() < 4:
            return {'uid': uid, 'band': band, 'snr': None,
                    'error': f'mask too small ({masked.count()} pixels)'}
        mask = np.ma.getmask(masked)

        m_no = np.ma.masked_array(exp_no_sub.data, mask=mask)
        m_yes = np.ma.masked_array(exp_with_sub.data, mask=mask)

        a = np.ma.compressed(m_yes)
        b = np.ma.compressed(m_no)
        if np.any(b == 0):
            return {'uid': uid, 'band': band, 'snr': None,
                    'error': 'zero in chi2 denominator'}

        chi2 = stats.chi_square(a, b)
        if not np.isfinite(chi2) or chi2 <= 0:
            return {'uid': uid, 'band': band, 'snr': None,
                    'error': f'bad chi2={chi2}'}

        return {'uid': uid, 'band': band, 'snr': float(np.sqrt(chi2)),
                'error': None}
    except Exception as e:
        return {'uid': uid, 'band': None, 'snr': None, 'error': repr(e)}


def plot_histogram(vals, out_png, band, pipeline_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = max(float(vals.min()), 0.1)
    hi = float(vals.max())
    if hi <= lo:
        bins = 40
    else:
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax.hist(vals, bins=bins)
    if isinstance(bins, np.ndarray):
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'SNR of substructure ($\sqrt{\chi^2}$)')
    ax.set_ylabel('Count')
    ax.set_title(f'{pipeline_label} — {band} (N={len(vals)})')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(args):
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.nice(19)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['data_dir']
    pipeline_label = cfg['pipeline_label']
    pipeline_dir = os.path.join(data_dir, pipeline_label)
    psf_cache_dir = os.path.join(data_dir, cfg['psf_cache_dir'])
    supersampling_factor = cfg['synthetic_image']['supersampling_factor']
    psf_num_pix = cfg['psf']['num_pixes'][0]
    exposure_time = cfg['imaging']['exposure_time']
    base_engine_params = cfg['imaging']['engine_params']
    snr_quantile = 0.95

    out_dir = os.path.join(pipeline_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'substructure_snr_failures.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_handler)

    workers = args.workers or max(1, multiprocessing.cpu_count() // 2)

    print(f'data_dir         = {data_dir}')
    print(f'pipeline_dir     = {pipeline_dir}')
    print(f'psf_cache_dir    = {psf_cache_dir}')
    print(f'out_dir          = {out_dir}')
    print(f'exposure_time    = {exposure_time}')
    print(f'bands            = {args.bands}')
    print(f'limit            = {args.limit}')
    print(f'workers          = {workers}')
    print()

    summary_rows = []
    for band in args.bands:
        synth_paths = discover_synth_pickles(pipeline_dir, band)
        if args.limit is not None and args.limit < len(synth_paths):
            synth_paths = synth_paths[:args.limit]
        if not synth_paths:
            print(f'[{band}] no SyntheticImage pickles found, skipping')
            continue
        print(f'[{band}] processing {len(synth_paths)} system(s) with {workers} worker(s)')

        tasks = [(p, exposure_time, base_engine_params,
                  psf_cache_dir, supersampling_factor, psf_num_pix, snr_quantile)
                 for p in synth_paths]

        start = time.time()
        results = []
        n_workers = min(workers, len(tasks))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_substructure_snr, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({'uid': '?', 'band': band, 'snr': None,
                                    'error': f'worker crash: {e!r}'})
        elapsed = time.time() - start

        snr_vals = np.array([r['snr'] for r in results if r['snr'] is not None],
                            dtype=float)
        uids = np.array([r['uid'] for r in results if r['snr'] is not None],
                        dtype=object)
        failures = {r['uid']: r['error'] for r in results
                    if r['snr'] is None and not r.get('skipped')}
        skipped = {r['uid']: r['error'] for r in results if r.get('skipped')}

        for uid, err in failures.items():
            logger.warning('[%s] uid=%s error=%s', band, uid, err)
        for uid, err in skipped.items():
            logger.info('[%s] uid=%s skipped: %s', band, uid, err)

        np.save(os.path.join(out_dir, f'substructure_snr_{band}.npy'), snr_vals)
        np.save(os.path.join(out_dir, f'substructure_snr_{band}_uids.npy'), uids)
        with open(os.path.join(out_dir, f'substructure_snr_failures_{band}.json'), 'w') as f:
            json.dump(failures, f, indent=2)

        if len(snr_vals):
            plot_histogram(snr_vals, os.path.join(out_dir, f'substructure_snr_hist_{band}.png'), band, pipeline_label)

        median = float(np.median(snr_vals)) if len(snr_vals) else float('nan')
        snr_max = float(np.max(snr_vals)) if len(snr_vals) else float('nan')
        summary_rows.append((band, len(snr_vals), len(failures), len(skipped),
                             median, snr_max, elapsed))

    print()
    print(f'{"band":<6} {"ok":>6} {"fail":>6} {"skip":>6} {"median":>10} {"max":>10} {"sec":>8}')
    for band, n_ok, n_fail, n_skip, med, mx, sec in summary_rows:
        print(f'{band:<6} {n_ok:>6d} {n_fail:>6d} {n_skip:>6d} {med:>10.3f} {mx:>10.3f} {sec:>8.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--bands', nargs='+', default=['F129'])
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    main(args)
