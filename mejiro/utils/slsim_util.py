import pandas as pd
from tqdm import tqdm


def write_lens_population_to_csv(output_path, lens_population, snr_list, verbose=False):
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
    verbose : bool, optional
        If True, prints progress and completion messages. Default is False.
    """
    # retrieve the bands
    sample_gglens = lens_population[0]
    bands = [k.split("_")[1] for k in sample_gglens.deflector._deflector._deflector_dict.keys() if k.startswith("mag_")]

    data = []
    for gglens, snr in tqdm(zip(lens_population, snr_list), total=len(lens_population), disable=not verbose):
        row = {
            'vel_disp': gglens.deflector_velocity_dispersion(),
            'm_star': gglens.deflector_stellar_mass(),
            'theta_e': gglens.einstein_radius[0],
            'z_lens': gglens.deflector_redshift,
            'z_source': gglens.source_redshift_list[0],
            'magnification': gglens.extended_source_magnification(),
            'num_images': len(gglens.point_source_image_positions()[0]),
            'snr': snr,
        }

        for band in bands:
            row[f'mag_{band}_lens'] = gglens.deflector_magnitude(band=band)
            row[f'mag_{band}_source'] = gglens.extended_source_magnitude(band=band, lensed=False)[0]
            row[f'mag_{band}_source_magnified'] = gglens.extended_source_magnitude(band=band, lensed=True)[0]

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    if verbose: print(f'Wrote lens population to {output_path}.')
