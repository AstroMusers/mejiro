import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import gammaincinv
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util as lenstronomy_util


def _sersic_b_n(n_sersic):
    """Sersic b_n parameter via the standard gammaincinv relation."""
    return float(gammaincinv(2.0 * n_sersic, 0.5))


def fit_sersic(image, pixel_scale, initial_kwargs=None, psf_kernel=None,
               fit_sky=False, use_poisson_weights=False, read_noise=0.05):
    """
    Fit a Sersic profile (optionally + sky pedestal) to a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D image array (e.g., exposure.data or synthetic_image.data).
    pixel_scale : float
        Pixel scale in arcsec/pix.
    initial_kwargs : dict, optional
        Initial guess for Sersic parameters. Keys: 'amp', 'R_sersic', 'n_sersic',
        'center_x', 'center_y', 'e1', 'e2', and optionally 'sky'. If None,
        defaults are derived from the image (amp from peak/exp(b_n), sky from
        corner-median).
    psf_kernel : np.ndarray, optional
        PSF kernel to convolve the model with before comparing to data.
    fit_sky : bool, optional
        If True, fit an additive constant ``sky`` pedestal alongside the
        Sersic. Strongly recommended for calibrated rate images (e.g., romanisim
        DN/s) where a uniform sky/dark floor would otherwise be absorbed into
        the Sersic shape, biasing R_sersic to large values. Default False to
        preserve backward compatibility on noiseless test inputs.
    use_poisson_weights : bool, optional
        If True, the squared-error cost is weighted by ``1 / (image + read_noise**2)``
        so that bright central pixels don't dominate the fit at the expense of
        the faint extended profile. Default False.
    read_noise : float, optional
        Noise floor added under the sqrt for the Poisson-weight denominator,
        in image units. Ignored when ``use_poisson_weights=False``. Default 0.05.

    Returns
    -------
    best_fit_kwargs : dict
        Best-fit Sersic parameters. Includes ``'sky'`` when ``fit_sky=True``.
    model_image : np.ndarray
        2D model image evaluated with the best-fit parameters. When
        ``fit_sky=True`` this is the Sersic + sky sum (so ``image - model_image``
        cleanly removes the lens and sky pedestal).
    fit_result : scipy.optimize.OptimizeResult
        Full optimization result from scipy.
    """
    num_pix = image.shape[0]
    fov = num_pix * pixel_scale

    x, y, *_ = lenstronomy_util.make_grid_with_coordtransform(
        numPix=num_pix, deltapix=pixel_scale, subgrid_res=1,
        left_lower=False, inverse=False)
    light_model = LightModel(['SERSIC_ELLIPSE'])

    # initial guesses
    if initial_kwargs is None:
        initial_kwargs = {}
    n_init = initial_kwargs.get('n_sersic', 4.0)
    # corner-median sky estimate; only used if fit_sky=True
    corner = min(8, num_pix // 4)
    sky_init = float(np.median(np.concatenate([
        image[:corner, :corner].ravel(),
        image[:corner, -corner:].ravel(),
        image[-corner:, :corner].ravel(),
        image[-corner:, -corner:].ravel(),
    ])))
    # peak-amp seed: for a SERSIC_ELLIPSE the central surface brightness is
    # amp * exp(b_n), so seeding amp = (peak - sky_init) / exp(b_n) puts the
    # initial central model flux near the observed peak.
    peak_above_sky = max(float(image.max()) - (sky_init if fit_sky else 0.0), 1e-6)
    amp_init = peak_above_sky / np.exp(_sersic_b_n(n_init))

    defaults = {
        'amp': amp_init,
        'R_sersic': 0.5,
        'n_sersic': n_init,
        'center_x': 0.0,
        'center_y': 0.0,
        'e1': 0.0,
        'e2': 0.0,
        'sky': sky_init,
    }
    for k, v in defaults.items():
        initial_kwargs.setdefault(k, v)

    data_flat = image.ravel()

    if use_poisson_weights:
        inv_var_flat = (1.0 / (np.clip(image, 0, None) + read_noise ** 2 + 1e-12)).ravel()
    else:
        inv_var_flat = None

    def _evaluate_sersic(params):
        amp, Rs, ns, cx, cy, e1, e2 = params[:7]
        kwargs = [{'amp': amp, 'R_sersic': Rs, 'n_sersic': ns,
                   'center_x': cx, 'center_y': cy, 'e1': e1, 'e2': e2}]
        m = light_model.surface_brightness(x, y, kwargs).reshape(num_pix, num_pix)
        if psf_kernel is not None:
            m = fftconvolve(m, psf_kernel, mode='same')
        return m

    def _model(params):
        m = _evaluate_sersic(params)
        if fit_sky:
            m = m + params[7]
        return m

    def cost(params):
        m_flat = _model(params).ravel()
        sq = (data_flat - m_flat) ** 2
        if inv_var_flat is not None:
            sq = sq * inv_var_flat
        return float(np.sum(sq))

    x0 = [initial_kwargs['amp'], initial_kwargs['R_sersic'],
          initial_kwargs['n_sersic'], initial_kwargs['center_x'],
          initial_kwargs['center_y'], initial_kwargs['e1'],
          initial_kwargs['e2']]
    bounds = [
        (1e-12, None),                # amp > 0
        (0.05, max(fov / 2.0, 0.1)),  # R_sersic ≤ FOV/2: prevents flat-pedestal degeneracy
        (0.5, 10.0),                  # n_sersic
        (-0.2, 0.2),                  # center_x
        (-0.2, 0.2),                  # center_y
        (-0.5, 0.5),                  # e1
        (-0.5, 0.5),                  # e2
    ]
    if fit_sky:
        x0.append(initial_kwargs['sky'])
        bounds.append((0, None))      # sky ≥ 0

    result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds)

    best_fit_kwargs = {
        'amp': result.x[0],
        'R_sersic': result.x[1],
        'n_sersic': result.x[2],
        'center_x': result.x[3],
        'center_y': result.x[4],
        'e1': result.x[5],
        'e2': result.x[6],
    }
    if fit_sky:
        best_fit_kwargs['sky'] = result.x[7]

    model_image = _model(result.x)

    return best_fit_kwargs, model_image, result


def subtract_lens(exposure, initial_kwargs=None, derive_initial_kwargs=False,
                  psf_kernel=None, fit_sky=False, use_poisson_weights=False,
                  read_noise=0.05):
    """
    Fit a Sersic profile to the central lensing galaxy in an Exposure and subtract it.

    Parameters
    ----------
    exposure : Exposure
        The exposure object to perform lens subtraction on.
    initial_kwargs : dict, optional
        Initial Sersic parameters for the fit. If provided, these are used
        directly and ``derive_initial_kwargs`` is ignored.
    derive_initial_kwargs : bool, optional
        If True and ``initial_kwargs`` is None, derive the initial Sersic guess
        from the strong lens model parameters stored on the Exposure
        (``exposure.synthetic_image.strong_lens.kwargs_lens_light``). If False
        (the default), no derivation occurs and :func:`fit_sersic` falls back to
        its built-in generic defaults.
    psf_kernel : np.ndarray, optional
        PSF kernel to convolve the model with during fitting. If None, no PSF
        convolution is applied.
    fit_sky : bool, optional
        Pass-through to :func:`fit_sersic`. Recommended for calibrated rate images.
    use_poisson_weights : bool, optional
        Pass-through to :func:`fit_sersic`.
    read_noise : float, optional
        Pass-through to :func:`fit_sersic`.

    Returns
    -------
    residual : np.ndarray
        Lens-subtracted image (``data - model``). When ``fit_sky=True`` the
        model includes the sky pedestal, so the residual is also sky-subtracted.
    model_image : np.ndarray
        Best-fit model image (Sersic, or Sersic + sky when ``fit_sky=True``).
    best_fit_kwargs : dict
        Best-fit parameters.
    fit_result : scipy.optimize.OptimizeResult
        Full optimization result from scipy.
    """
    image = exposure.data
    pixel_scale = exposure.synthetic_image.pixel_scale

    if initial_kwargs is None and derive_initial_kwargs:
        strong_lens = exposure.synthetic_image.strong_lens
        lens_light_kwargs = strong_lens.kwargs_lens_light[0]

        initial_kwargs = {
            'amp': np.max(image),
            'R_sersic': lens_light_kwargs.get('R_sersic', 0.5),
            'n_sersic': lens_light_kwargs.get('n_sersic', 4.0),
            'center_x': lens_light_kwargs.get('center_x', 0.0),
            'center_y': lens_light_kwargs.get('center_y', 0.0),
            'e1': lens_light_kwargs.get('e1', 0.0),
            'e2': lens_light_kwargs.get('e2', 0.0),
        }

    best_fit_kwargs, model_image, fit_result = fit_sersic(
        image, pixel_scale, initial_kwargs=initial_kwargs, psf_kernel=psf_kernel,
        fit_sky=fit_sky, use_poisson_weights=use_poisson_weights, read_noise=read_noise)

    residual = image - model_image

    return residual, model_image, best_fit_kwargs, fit_result


def plot_lens_subtraction(image, model, residual, savepath=None):
    """
    Plot the original image, fitted Sersic model, and lens-subtracted residual.

    Parameters
    ----------
    image, model, residual : np.ndarray
        2D arrays to display.
    savepath : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    combined = np.concatenate([image[image > 0], model[model > 0]])
    vmin = float(combined.min())
    vmax = float(max(image.max(), model.max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cbar_label = 'Counts (log scale)'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(image, origin='lower', cmap='viridis', norm=norm)
    axes[0].set_title('Original')
    fig.colorbar(im0, ax=axes[0], label=cbar_label)

    im1 = axes[1].imshow(model, origin='lower', cmap='viridis', norm=norm)
    axes[1].set_title('Fitted Sersic Model')
    fig.colorbar(im1, ax=axes[1], label=cbar_label)

    rmax = float(np.max(np.abs(residual)))
    im2 = axes[2].imshow(residual, origin='lower', cmap='bwr', vmin=-rmax, vmax=rmax)
    axes[2].set_title('Residual (Data $-$ Model)')
    fig.colorbar(im2, ax=axes[2], label='Counts')

    for ax in axes:
        ax.set_xlabel('x [Pixels]')
        ax.set_ylabel('y [Pixels]')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()
