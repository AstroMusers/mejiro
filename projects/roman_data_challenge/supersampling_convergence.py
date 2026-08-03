"""Measure how ``synthetic_image.supersampling_factor`` converges, and whether the value the
configs ship is small enough to be harmless.

At ``oversample > 1`` the step-04 grid is already finer than a detector pixel, so
``supersampling_factor`` subdivides a subpixel rather than a detector pixel and its right
value has to be chosen on its own terms (see docs/step04_oversampled_rendering.md). The
rung-1 configs use 3, picked against a factor-5 reference on three lenses. This script puts a
figure behind that: it renders a sample of real systems across a ladder of factors, reduces
each to detector resolution the way ``_05_romanisim`` does, and plots the residual error
against the substructure signal it has to stay below.

The comparison happens on the *binned* arrays because detector resolution is all step 05 ever
consumes; sub-pixel detail that binning averages away is not error anyone sees.

Every render pins the detector position via the pipeline's own ``_detector_position``, so all
factors see the same STPSF kernel. That is not a nicety: the draw used to be an unseeded
``random.choice``, and the first attempt at this measurement compared different PSFs rather
than different factors.

Read-only with respect to the pipeline. Reads ``03/`` lens pickles and writes only under its
own output directory; it never enters a step directory and never calls a pipeline ``main()``.

Usage:
    python3 supersampling_convergence.py --config <config.yaml>
        [--band F129] [--factors 1 2 3 5 7 9] [--n-lenses 12] [--oversample 5]
        [--workers N] [--output-dir DIR] [--resume]

Outputs (under ``<data_dir>/<pipeline_label>/analysis/supersampling_convergence/`` by default):
    cache/<uid>_<band>_ov<O>_ssf<F>.npy    detector-resolution render, one per (lens, factor)
    cache/<uid>_<band>_ov<O>_nosub.npy     no-substructure counterpart at the top factor
    supersampling_convergence.json         every number behind the figure
    supersampling_convergence.csv          the same, flat
    supersampling_convergence.png / .pdf   the figure
"""
import os

# Keep BLAS single-threaded so N workers do not collectively oversubscribe the host, the same
# way the pipeline scripts do. Must precede the numpy import.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import argparse
import csv
import json
import logging
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Pixels fainter than this fraction of the peak are excluded from the median metric: their
# fractional error is dominated by the sky-level tail and says nothing about the flux that
# matters.
FLOOR_FRACTION = 0.01

NOSUB = 'nosub'


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def _render(task):
    """Render one (lens, factor) and return its detector-resolution array plus timings.

    Runs in a worker process, so every mejiro import is local and the mejiro-v2 pickle shim
    is re-applied here -- a spawn worker inherits neither.
    """
    (lens_path, variant, supersampling_factor, uid, band, oversample, compute_mode,
     fov_arcsec, psf_config, seed, cache_path, psf_cache_dir) = task

    from mejiro.instruments.roman import Roman
    from mejiro.pipeline._04_create_synthetic_images import _detector_position
    from mejiro.pipeline._05_romanisim import bin_to_native
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import roman_util, util
    from mejiro.utils.pipeline_helper import PipelineHelper

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()

    lens = util.unpickle(lens_path)
    detector = int(os.path.basename(os.path.dirname(lens_path))[3:])
    position = _detector_position(lens.name, seed,
                                  roman_util.divide_up_sca(psf_config['divide_up_detector']))
    instrument_params = {'detector': detector, 'detector_position': position}

    if variant == NOSUB:
        # Swap the active lens model for the macromodel backups add_realization() saved, as
        # in substructure_snr_histogram.make_no_substructure_synth.
        for attr in ('kwargs_lens_macromodel', 'lens_model_list_macromodel',
                     'lens_redshift_list_macromodel'):
            if getattr(lens, attr, None) is None:
                raise RuntimeError(f'{attr} missing on {lens.name}; it carries no realization')
        lens.kwargs_lens = deepcopy(lens.kwargs_lens_macromodel)
        lens.lens_model_list = deepcopy(lens.lens_model_list_macromodel)
        lens.lens_redshift_list = deepcopy(lens.lens_redshift_list_macromodel)
        lens.realization = None
        lens.use_jax = list(lens.use_jax)[:len(lens.lens_model_list)]

    instrument = Roman()
    kwargs_psf = instrument.get_psf_kwargs(
        band=band, detector=detector, detector_position=position,
        oversample=oversample, num_pix=psf_config['num_pixes'][0],
        check_cache=True, psf_cache_dir=psf_cache_dir, require_cached=True,
        degrade=oversample == 1,
    )

    start = time.time()
    synth = SyntheticImage(
        strong_lens=lens,
        instrument=instrument,
        band=band,
        fov_arcsec=fov_arcsec,
        instrument_params=instrument_params,
        # A FRESH dict per construction: SyntheticImage caches supersampled_indexes into the
        # caller's kwargs_numerics, and reusing one across grid sizes reuses the wrong mask.
        kwargs_numerics={'supersampling_factor': supersampling_factor,
                         'compute_mode': compute_mode},
        kwargs_psf=kwargs_psf,
        pieces=False,
        oversample=oversample,
    )
    elapsed = time.time() - start

    binned = bin_to_native(np.asarray(synth.data, dtype=np.float64), oversample)

    tmp = cache_path + '.tmp.npy'
    np.save(tmp, binned)
    os.replace(tmp, cache_path)

    return {
        'uid': uid, 'variant': variant, 'seconds': elapsed,
        'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        'detector_position': list(position),
    }


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def bright_mask(reference):
    """Pixels worth measuring: fainter ones have fractional errors dominated by the sky tail."""
    return reference > FLOOR_FRACTION * reference.max()


def signal_mask(with_sub, no_sub, percentile=90):
    """Pixels where substructure actually shows up.

    The median over all bright pixels is dominated by the deflector core, which substructure
    barely touches, so it understates how the numerics compare where the signal lives. This
    restricts to the brightest decile of the |with - without| difference.
    """
    bright = bright_mask(with_sub)
    diff = np.abs(with_sub - no_sub)
    return bright & (diff >= np.percentile(diff[bright], percentile))


def compare(a, b, mask=None):
    """Error of ``a`` relative to reference ``b``, both at detector resolution."""
    if mask is None:
        mask = bright_mask(b)
    diff = a - b
    return {
        'median_frac_err': float(np.median(np.abs(diff[mask]) / b[mask])),
        'max_over_peak': float(np.abs(diff).max() / b.max()),
        'peak_ratio': float(a.max() / b.max()),
        'flux_ratio': float(a.sum() / b.sum()),
    }


# --------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------

def make_figure(results, factors, top_factor, output_base):
    from mejiro import style
    style.set_aas_style()

    uids = sorted(results['per_lens'])
    ladder = [f for f in factors if f != top_factor]

    fig, axes = plt.subplots(1, 3, figsize=(style.TWO_COLUMN_WIDTH, 2.6))

    def series(uid, key, metric):
        return [results['per_lens'][uid][key].get(str(f), {}).get(metric, np.nan)
                for f in ladder]

    # --- (a) absolute error, with the substructure signal for scale --------------------
    ax = axes[0]
    for uid in uids:
        ax.plot(ladder, series(uid, 'vs_top', 'median_frac_err'), color='0.75', lw=0.6, zorder=1)
    ax.plot(ladder, [np.nanmedian([results['per_lens'][u]['vs_top'].get(str(f), {}).get(
        'median_frac_err', np.nan) for u in uids]) for f in ladder],
        'o-', color='#0C5DA5', lw=1.6, ms=3.5, zorder=3, label='median over lenses')

    signal = [results['per_lens'][u]['signal']['median_frac_err']
              for u in uids if results['per_lens'][u].get('signal')]
    if signal:
        ax.axhspan(np.min(signal), np.max(signal), color='#FF9500', alpha=0.18, zorder=0)
        ax.axhline(np.median(signal), color='#FF9500', lw=1.2, zorder=2,
                   label='substructure signal')
    ax.set_yscale('log')
    ax.set_xlabel('supersampling_factor')
    ax.set_ylabel('median $|\\Delta| \\, / \\,$ value')
    ax.set_title(f'(a) numerical error\n(pixels $>${FLOOR_FRACTION:.0%} of peak)', fontsize=8)
    ax.axvline(3, color='0.4', ls='--', lw=0.8, zorder=0)

    # --- (b) the sufficiency test: error as a fraction of the signal it could mimic ----
    ax = axes[1]
    ratios = {}
    for uid in uids:
        entry = results['per_lens'][uid]
        if not entry.get('signal_region'):
            continue
        denom = entry['signal_region']['median_frac_err']
        ratios[uid] = [entry['vs_top_signal_region'].get(str(f), {}).get(
            'median_frac_err', np.nan) / denom for f in ladder]
        ax.plot(ladder, ratios[uid], color='0.75', lw=0.6, zorder=1)
    if ratios:
        ax.plot(ladder, np.nanmedian(np.array(list(ratios.values())), axis=0),
                'o-', color='#0C5DA5', lw=1.6, ms=3.5, zorder=3)
    ax.axhline(1.0, color='#FF9500', lw=1.2, zorder=2)
    ax.text(0.97, 0.93, 'error = signal', transform=ax.transAxes, ha='right', va='top',
            fontsize=6, color='#FF9500')
    ax.set_yscale('log')
    ax.set_xlabel('supersampling_factor')
    ax.set_ylabel('numerical error $/$ substructure signal')
    ax.set_title('(b) sufficiency\n(pixels carrying the signal)', fontsize=8)
    ax.axvline(3, color='0.4', ls='--', lw=0.8, zorder=0)

    ax = axes[2]
    for uid in uids:
        ax.plot(factors, [results['per_lens'][uid]['seconds'].get(str(f), np.nan) for f in factors],
                color='0.75', lw=0.6)
    ax.plot(factors, [np.nanmedian([results['per_lens'][u]['seconds'].get(str(f), np.nan)
                                    for u in uids]) for f in factors],
            'o-', color='#00B945', lw=1.6, ms=3.5)
    ax.set_yscale('log')
    ax.set_xlabel('supersampling_factor')
    ax.set_ylabel('seconds / lens-band')
    ax.set_title('(c) cost', fontsize=8)
    ax.axvline(3, color='0.4', ls='--', lw=0.8, zorder=0)

    axes[0].legend(fontsize=6, frameon=False, loc='best')
    fig.suptitle(
        f"step-04 supersampling convergence at oversample={results['oversample']}, "
        f"{results['band']}, {len(uids)} systems (reference: factor {top_factor})",
        fontsize=8, y=1.04,
    )
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{output_base}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Wrote {output_base}.png and {output_base}.pdf')


# --------------------------------------------------------------------------------------

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s', force=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    si_config, psf_config = config['synthetic_image'], config['psf']
    band = args.band or si_config['bands'][0]
    oversample = args.oversample if args.oversample is not None else si_config.get('oversample', 1)
    compute_mode = si_config['supersampling_compute_mode']
    fov_arcsec = si_config['fov_arcsec']
    seed = config['seed']

    factors = sorted(args.factors)
    top_factor = factors[-1]

    pipeline_dir = os.path.join(config['data_dir'], config['pipeline_label'])
    output_dir = args.output_dir or os.path.join(pipeline_dir, 'analysis',
                                                 'supersampling_convergence')
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    psf_cache_dir = os.path.join(config['data_dir'], config['psf_cache_dir'])

    # --- select systems ---------------------------------------------------------------
    lens_paths = sorted(glob(os.path.join(pipeline_dir, '03', 'sca*', 'lens_*.pkl')))
    if not lens_paths:
        raise FileNotFoundError(f'No step-03 lens pickles under {pipeline_dir}/03')
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(lens_paths, min(args.n_lenses, len(lens_paths)),
                               replace=False).tolist())
    logger.info(f'Sampled {len(chosen)} of {len(lens_paths)} step-03 systems, band {band}, '
                f'oversample {oversample}, compute_mode {compute_mode!r}')
    logger.info(f'Factor ladder: {factors} (reference: {top_factor})')

    def uid_of(path):
        return os.path.basename(path)[len('lens_'):-len('.pkl')]

    def cache_path(uid, factor):
        tag = NOSUB if factor == NOSUB else f'ssf{factor}'
        return os.path.join(cache_dir, f'{uid}_{band}_ov{oversample}_{tag}.npy')

    def nosub_missing_path(uid):
        """Marker for a system with no realization, so --resume stops re-attempting it.

        The macromodel check happens after the (slow) unpickle, so without this a system
        that can never produce a signal reference costs a load on every resumed run.
        """
        return cache_path(uid, NOSUB) + '.missing'

    # Timings are the one result not recoverable from the cached arrays, so they are
    # persisted and merged rather than recomputed; otherwise a --resume run plots nothing.
    timings_path = os.path.join(cache_dir, 'timings.json')
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings = json.load(f)
    else:
        timings = {}

    # --- build tasks, most expensive first so the long tail starts early ---------------
    tasks = []
    for lens_path in chosen:
        uid = uid_of(lens_path)
        for variant in list(factors) + [NOSUB]:
            path = cache_path(uid, variant)
            if args.resume and os.path.exists(path):
                continue
            if variant == NOSUB and os.path.exists(nosub_missing_path(uid)):
                continue
            # the no-substructure counterpart is rendered at the reference factor, so the
            # signal it defines is not itself limited by supersampling
            ssf = top_factor if variant == NOSUB else variant
            tasks.append((lens_path, variant, ssf, uid, band, oversample, compute_mode,
                          fov_arcsec, psf_config, seed, path, psf_cache_dir))
    tasks.sort(key=lambda t: t[2], reverse=True)

    workers = args.workers or min(config['cores']['script_04'], os.cpu_count())
    logger.info(f'{len(tasks)} render(s) to run with {workers} worker(s)')

    positions, nosub_failures, peak_rss = {}, {}, 0.0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures)):
                task = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    if task[1] == NOSUB:
                        # A system with no realization has no macromodel backup; it simply
                        # contributes no signal reference. Record it so --resume moves on.
                        nosub_failures[task[3]] = str(e)
                        with open(nosub_missing_path(task[3]), 'w') as fh:
                            fh.write(str(e))
                        continue
                    raise
                timings.setdefault(r['uid'], {})[str(r['variant'])] = r['seconds']
                peak_rss = max(peak_rss, r['peak_rss_gb'])
                # every factor must have seen the same PSF -- the confound this study exists
                # to avoid
                prev = positions.setdefault(r['uid'], r['detector_position'])
                assert prev == r['detector_position'], (
                    f"detector position moved between renders of {r['uid']}: "
                    f"{prev} vs {r['detector_position']}"
                )
        if peak_rss:
            logger.info(f'Peak RSS in any single worker: {peak_rss:.2f} GB '
                        f'(x{workers} workers; scales as factor^2)')
        with open(timings_path, 'w') as f:
            json.dump(timings, f, indent=2)
    if nosub_failures:
        logger.info(f'{len(nosub_failures)} system(s) carry no realization; '
                    f'they contribute no substructure-signal reference')

    # --- metrics ----------------------------------------------------------------------
    results = {
        'config': os.path.abspath(args.config), 'band': band, 'oversample': oversample,
        'compute_mode': compute_mode, 'factors': factors, 'top_factor': top_factor,
        'floor_fraction': FLOOR_FRACTION, 'n_lenses': len(chosen), 'per_lens': {},
    }
    for lens_path in chosen:
        uid = uid_of(lens_path)
        arrays = {f: np.load(cache_path(uid, f)) for f in factors}
        entry = {'vs_top': {}, 'vs_previous': {}, 'vs_top_signal_region': {},
                 'signal': None, 'signal_region': None, 'seconds': timings.get(uid, {})}

        for i, factor in enumerate(factors):
            if factor != top_factor:
                entry['vs_top'][str(factor)] = compare(arrays[factor], arrays[top_factor])
            if i:
                entry['vs_previous'][str(factor)] = compare(arrays[factor], arrays[factors[i - 1]])

        nosub = cache_path(uid, NOSUB)
        if os.path.exists(nosub):
            # substructure signal: with minus without, at the converged factor
            no_sub = np.load(nosub)
            entry['signal'] = compare(arrays[top_factor], no_sub)
            # ...and the same, restricted to where substructure actually shows up, together
            # with the numerical error measured over those same pixels. The ratio of the two
            # is the sufficiency test.
            region = signal_mask(arrays[top_factor], no_sub)
            entry['signal_region'] = compare(arrays[top_factor], no_sub, mask=region)
            for factor in factors:
                if factor != top_factor:
                    entry['vs_top_signal_region'][str(factor)] = compare(
                        arrays[factor], arrays[top_factor], mask=region)
        results['per_lens'][uid] = entry

    base = os.path.join(output_dir, 'supersampling_convergence')
    with open(base + '.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'comparison', 'factor', 'median_frac_err', 'max_over_peak',
                    'peak_ratio', 'flux_ratio', 'seconds'])
        for uid, entry in sorted(results['per_lens'].items()):
            for kind in ('vs_top', 'vs_previous'):
                for factor, m in sorted(entry[kind].items(), key=lambda kv: int(kv[0])):
                    w.writerow([uid, kind, factor, m['median_frac_err'], m['max_over_peak'],
                                m['peak_ratio'], m['flux_ratio'], entry['seconds'].get(factor, '')])
            if entry['signal']:
                s = entry['signal']
                w.writerow([uid, 'substructure_signal', top_factor, s['median_frac_err'],
                            s['max_over_peak'], s['peak_ratio'], s['flux_ratio'], ''])

    make_figure(results, factors, top_factor, base)

    # --- verdict ----------------------------------------------------------------------
    per_lens = list(results['per_lens'].values())
    with_signal = [e for e in per_lens if e.get('signal_region')]

    print(f'\nConvergence vs supersampling_factor {top_factor}, median over {len(per_lens)} systems')
    print(f'{"factor":>7}  {"median |d|/val":>15}  {"max |d|/peak":>13}  {"err/signal":>11}  {"s/lens-band":>12}')
    for factor in factors[:-1]:
        med = np.nanmedian([e['vs_top'][str(factor)]['median_frac_err'] for e in per_lens])
        mx = np.nanmedian([e['vs_top'][str(factor)]['max_over_peak'] for e in per_lens])
        sec = np.nanmedian([e['seconds'].get(str(factor), np.nan) for e in per_lens])
        ratio = np.nanmedian([
            e['vs_top_signal_region'][str(factor)]['median_frac_err']
            / e['signal_region']['median_frac_err'] for e in with_signal
        ]) if with_signal else float('nan')
        print(f'{factor:>7}  {med:>15.5f}  {mx:>13.5f}  {ratio:>11.3f}  {sec:>12.1f}')

    if with_signal:
        print(f'\nSufficiency, over the pixels carrying the substructure signal '
              f'({len(with_signal)} of {len(per_lens)} systems have a realization):')
        for factor in factors[:-1]:
            ratio = np.nanmedian([
                e['vs_top_signal_region'][str(factor)]['median_frac_err']
                / e['signal_region']['median_frac_err'] for e in with_signal
            ])
            verdict = (f'numerics are {1 / ratio:.0f}x smaller than the signal' if ratio < 1
                       else 'NUMERICS EXCEED THE SIGNAL')
            print(f'  factor {factor}: error/signal = {ratio:.3f}  -> {verdict}')
    else:
        print('\nNo sampled system carried a realization, so there is no signal reference. '
              'Raise --n-lenses.')
    print(f'\nWrote {base}.json/.csv/.png/.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Measure step-04 supersampling_factor convergence against the substructure signal.')
    parser.add_argument('--config', required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--band', default=None, help='Band to render (default: first of synthetic_image.bands).')
    parser.add_argument('--factors', type=int, nargs='+', default=[1, 2, 3, 5, 7, 9],
                        help='supersampling_factor ladder; the largest is the reference.')
    parser.add_argument('--n-lenses', dest='n_lenses', type=int, default=12,
                        help='Number of step-03 systems to sample.')
    parser.add_argument('--oversample', type=int, default=None,
                        help='Render oversample (default: synthetic_image.oversample).')
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker processes (default: cores.script_04, capped at cpu_count). '
                             'Cut this if raising --factors past 9: peak memory grows as factor^2.')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                        help='Where to write cache, tables and figure.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip renders whose cached array already exists.')
    main(parser.parse_args())
