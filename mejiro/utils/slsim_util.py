import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def write_lens_population_to_csv(output_path, lens_population, snr_list, bands=None, show_progress_bar=False):
    """
    Write list of SLSim galaxy-galaxy Lens objects to a CSV.

    Parameters
    ----------
    output_path : str
        The file path where the CSV file will be saved.
    lens_population : list
        A list of SLSim galaxy-galaxy Lens objects.
    snr_list : list
        A list of signal-to-noise ratio (SNR) values corresponding to each system.
    bands: list
        A list of bands. If None, they will be parsed from the SLSim Deflector.
    show_progress_bar : bool, optional
        If True, shows a progress bar during processing. Default is False.
    """
    logger.info(f'Writing lens population to {output_path}...')

    # retrieve the bands
    if bands is None:
        sample_gglens = lens_population[0]
        bands = [k.split("_")[1] for k in sample_gglens.deflector._deflector._deflector_dict.keys() if k.startswith("mag_")]

    data = []
    for gglens, snr in tqdm(zip(lens_population, snr_list), total=len(lens_population), disable=not show_progress_bar):
        row = {
            'vel_disp': gglens.deflector_velocity_dispersion(),
            'm_star': gglens.deflector_stellar_mass(),
            'theta_e': gglens.einstein_radius[0],
            'z_lens': gglens.deflector_redshift,
            'z_source': gglens.source_redshift_list[0],
            'magnification': gglens.extended_source_magnification[0],
            'num_images': len(gglens.extended_source_image_positions[0]),
            'snr': snr,
        }

        if gglens.deflector.deflector_type == "NFW_HERNQUIST":
            row['main_halo_mass'] = gglens.deflector.halo_properties[0]
            row['main_halo_concentration'] = gglens.deflector.halo_properties[1]

        for band in bands:
            row[f'mag_{band}_lens'] = gglens.deflector_magnitude(band=band)
            row[f'mag_{band}_source'] = gglens.extended_source_magnitude(band=band, lensed=False)[0]
            row[f'mag_{band}_source_magnified'] = gglens.extended_source_magnitude(band=band, lensed=True)[0]

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f'Wrote lens population to {output_path}.')


def remap_source_images(strong_lens, mapping):
    """
    Override entries in ``strong_lens.kwargs_params['source_images']`` by
    copying one band's source-image array into another band's slot.

    The dict ``source_images`` is populated by ``GalaxyGalaxy.from_slsim``
    (and equivalents) for catalog sources that ship per-band pixelated
    cutouts (e.g. SLSim's COSMOS-Web). Whichever per-band image is sitting
    under a key at construction time defines the morphology that
    ``SyntheticImage`` will ray-shoot for that band. This helper lets the
    caller swap which cutout backs which band — useful for experimenting
    with alternative catalog→instrument band mappings without modifying
    the underlying catalog code.

    Mutates ``strong_lens.kwargs_params['source_images']`` in place.

    Parameters
    ----------
    strong_lens : mejiro.strong_lens.StrongLens
        StrongLens (e.g. GalaxyGalaxy, LensedSupernova) built from a SLSim
        catalog source. Must have a populated
        ``kwargs_params['source_images']`` dict.
    mapping : dict[str, str]
        ``{destination_band: source_band}``. Both keys must be present in
        ``source_images`` *before* the call. After the call,
        ``source_images[destination_band]`` is the array previously stored
        under ``source_images[source_band]``. Self-mappings are no-ops.
    """
    source_images = strong_lens.kwargs_params.get('source_images')
    if not source_images:
        raise ValueError(
            "strong_lens.kwargs_params has no 'source_images' entry; "
            "remap_source_images only applies to lenses built from a "
            "catalog source (e.g. SLSim COSMOS-Web)."
        )
    snapshot = dict(source_images)
    for dest_band, src_band in mapping.items():
        if src_band not in snapshot:
            raise KeyError(
                f"Source band '{src_band}' (for destination '{dest_band}') "
                f"not present in source_images. Available keys: {sorted(snapshot)}"
            )
        if dest_band not in source_images:
            raise KeyError(
                f"Destination band '{dest_band}' not present in source_images. "
                f"Available keys: {sorted(source_images)}"
            )
        source_images[dest_band] = snapshot[src_band]
