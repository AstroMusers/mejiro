"""Time LensModel.ray_shooting for one backend in an isolated process.

Driven from examples/jaxtronomy_pipeline_walkthrough.ipynb. JAX locks its
platform on first import, so each backend needs its own process; the caller
sets JAX_PLATFORM_NAME in the env before launching this script and selects
the backend via --backend.

Emits a single JSON line on stdout:
    {"backend": ..., "mean_s": ..., "std_s": ..., "n_trials": ..., "n_points": ...}
"""
import argparse
import json
import pickle
import sys
import time

import numpy as np


def _build_grid(fov_arcsec, n_pix):
    r = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, n_pix)
    xx, yy = np.meshgrid(r, r)
    return xx.ravel(), yy.ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['numpy', 'jax-cpu', 'jax-gpu'])
    parser.add_argument('--lens-pickle', required=True)
    parser.add_argument('--n-warmup', type=int, default=3)
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--n-pix', type=int, default=100)
    parser.add_argument('--fov-arcsec', type=float, default=5.0)
    args = parser.parse_args()

    with open(args.lens_pickle, 'rb') as f:
        lens = pickle.load(f)

    use_jax_flag = args.backend != 'numpy'
    lens.use_jax = [use_jax_flag] * len(lens.lens_model_list)

    lens_model = lens.lens_model
    kwargs_lens = lens.kwargs_lens
    x, y = _build_grid(args.fov_arcsec, args.n_pix)

    if use_jax_flag:
        import jax
        actual = jax.default_backend()
        requested = 'gpu' if args.backend == 'jax-gpu' else 'cpu'
        if requested == 'gpu' and actual != 'gpu':
            raise RuntimeError(
                f'jax-gpu requested but JAX defaulted to {actual!r}. '
                f'devices: {jax.devices()}'
            )

        def _run():
            bx, by = lens_model.ray_shooting(x, y, kwargs_lens)
            jax.block_until_ready(bx)
            jax.block_until_ready(by)
    else:
        def _run():
            lens_model.ray_shooting(x, y, kwargs_lens)

    for _ in range(args.n_warmup):
        _run()

    samples = np.empty(args.n_trials)
    for i in range(args.n_trials):
        t0 = time.perf_counter()
        _run()
        samples[i] = time.perf_counter() - t0

    result = {
        'backend': args.backend,
        'mean_s': float(samples.mean()),
        'std_s': float(samples.std(ddof=1)) if args.n_trials > 1 else 0.0,
        'n_trials': args.n_trials,
        'n_points': int(x.size),
    }
    print(json.dumps(result))


if __name__ == '__main__':
    sys.exit(main())
