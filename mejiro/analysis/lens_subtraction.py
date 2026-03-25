import numpy as np
from scipy.optimize import minimize
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util as lenstronomy_util


def fit_sersic(image, pixel_scale, initial_kwargs=None, psf_kernel=None):
    """
    Fit a Sersic profile to a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D image array (e.g., exposure.data or synthetic_image.data).
    pixel_scale : float
        Pixel scale in arcsec/pix.
    initial_kwargs : dict, optional
        Initial guess for Sersic parameters. Keys: 'amp', 'R_sersic', 'n_sersic',
        'center_x', 'center_y', 'e1', 'e2'. If None, defaults are used.
    psf_kernel : np.ndarray, optional
        PSF kernel to convolve the model with before comparing to data.

    Returns
    -------
    best_fit_kwargs : dict
        Best-fit Sersic parameters.
    model_image : np.ndarray
        2D model image evaluated with the best-fit parameters.
    fit_result : scipy.optimize.OptimizeResult
        Full optimization result from scipy.
    """
    num_pix = image.shape[0]

    # set up lenstronomy coordinate grid
    x, y, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
        lenstronomy_util.make_grid_with_coordtransform(
            numPix=num_pix,
            deltapix=pixel_scale,
            subgrid_res=1,
            left_lower=False,
            inverse=False))

    light_model = LightModel(['SERSIC_ELLIPSE'])

    # set initial guess
    if initial_kwargs is None:
        initial_kwargs = {
            'amp': np.max(image),
            'R_sersic': 0.5,
            'n_sersic': 4.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'e1': 0.0,
            'e2': 0.0,
        }

    data_flat = image.ravel()

    def _evaluate_model(params):
        amp, R_sersic, n_sersic, center_x, center_y, e1, e2 = params
        kwargs = [{'amp': amp, 'R_sersic': R_sersic, 'n_sersic': n_sersic,
                   'center_x': center_x, 'center_y': center_y,
                   'e1': e1, 'e2': e2}]
        model = light_model.surface_brightness(x, y, kwargs)
        if psf_kernel is not None:
            from scipy.signal import fftconvolve
            model_2d = model.reshape(num_pix, num_pix)
            model_2d = fftconvolve(model_2d, psf_kernel, mode='same')
            return model_2d.ravel()
        return model

    def cost(params):
        model = _evaluate_model(params)
        return np.sum((data_flat - model) ** 2)

    x0 = [initial_kwargs['amp'], initial_kwargs['R_sersic'],
           initial_kwargs['n_sersic'], initial_kwargs['center_x'],
           initial_kwargs['center_y'], initial_kwargs['e1'],
           initial_kwargs['e2']]

    fov = num_pix * pixel_scale / 2.0
    bounds = [
        (0, None),           # amp
        (0.01, 10.0),        # R_sersic
        (0.5, 10.0),         # n_sersic
        (-fov, fov),         # center_x
        (-fov, fov),         # center_y
        (-0.5, 0.5),         # e1
        (-0.5, 0.5),         # e2
    ]

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

    # generate model image
    model_flat = _evaluate_model(result.x)
    model_image = model_flat.reshape(num_pix, num_pix)

    return best_fit_kwargs, model_image, result


def subtract_lens(exposure, initial_kwargs=None, psf_kernel=None):
    """
    Fit a Sersic profile to the central lensing galaxy in an Exposure and subtract it.

    Parameters
    ----------
    exposure : Exposure
        The exposure object to perform lens subtraction on.
    initial_kwargs : dict, optional
        Initial Sersic parameters for the fit. If None, initial guesses are derived
        from the strong lens model parameters stored in the Exposure.
    psf_kernel : np.ndarray, optional
        PSF kernel to convolve the model with during fitting. If None, no PSF
        convolution is applied.

    Returns
    -------
    residual : np.ndarray
        Lens-subtracted image (data - model).
    model_image : np.ndarray
        Best-fit Sersic model image.
    best_fit_kwargs : dict
        Best-fit Sersic parameters.
    fit_result : scipy.optimize.OptimizeResult
        Full optimization result from scipy.
    """
    image = exposure.data
    pixel_scale = exposure.synthetic_image.pixel_scale

    # derive initial guesses from the strong lens if not provided
    if initial_kwargs is None:
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
        image, pixel_scale, initial_kwargs=initial_kwargs, psf_kernel=psf_kernel)

    residual = image - model_image

    return residual, model_image, best_fit_kwargs, fit_result


def plot_lens_subtraction(image, model, residual, savepath=None):
    """
    Plot the original image, fitted Sersic model, and lens-subtracted residual.

    Parameters
    ----------
    image : np.ndarray
        Original 2D image (e.g., exposure.data).
    model : np.ndarray
        Best-fit Sersic model image.
    residual : np.ndarray
        Residual image (data - model).
    savepath : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # original
    im0 = axes[0].imshow(np.log10(np.clip(image, 1, None)), origin='lower', cmap='viridis')
    axes[0].set_title('Original')
    fig.colorbar(im0, ax=axes[0], label=r'log$_{10}$(Counts)')

    # model
    im1 = axes[1].imshow(np.log10(np.clip(model, 1, None)), origin='lower', cmap='viridis')
    axes[1].set_title('Fitted Sersic Model')
    fig.colorbar(im1, ax=axes[1], label=r'log$_{10}$(Counts)')

    # residual
    vmax = np.max(np.abs(residual))
    im2 = axes[2].imshow(residual, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Residual (Data $-$ Model)')
    fig.colorbar(im2, ax=axes[2], label='Counts')

    for ax in axes:
        ax.set_xlabel('x [Pixels]')
        ax.set_ylabel('y [Pixels]')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()
