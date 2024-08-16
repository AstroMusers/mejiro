#!/usr/bin/env python
# coding: utf-8

import os

import slsim
import speclite
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Observations.roman_speclite import configure_roman_filters
from slsim.Observations.roman_speclite import filter_names
from slsim.lens_pop import LensPop
from tqdm import tqdm

from mejiro.helpers import survey_sim

path = os.path.dirname(slsim.__file__)
module_path, _ = os.path.split(path)
skypy_config = os.path.join(module_path, "data/SkyPy/roman-like.yml")

configure_roman_filters()

roman_filters = filter_names()

_ = speclite.filters.load_filters(
    roman_filters[0],
    roman_filters[1],
    roman_filters[2],
    roman_filters[3],
    roman_filters[4],
    roman_filters[5],
    roman_filters[6],
    roman_filters[7],
)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

sky_area = Quantity(value=5., unit="deg2")

deflector_mag_cuts = [20, 22, 24, 26, 28]  #
source_mag_cuts = [21, 23, 25, 27, 29]  # 

detectable_counts, lens_pop_counts = [], []

for deflector_mag_cut, source_mag_cut in tqdm(zip(deflector_mag_cuts, source_mag_cuts), total=len(deflector_mag_cuts)):

    kwargs_deflector_cut = {"band": "F062", "band_max": deflector_mag_cut, "z_min": 0.01, "z_max": 3.0}
    kwargs_source_cut = {"band": "F062", "band_max": source_mag_cut, "z_min": 0.01, "z_max": 5.0}

    lens_pop = LensPop(
        deflector_type="all-galaxies",
        source_type="galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        kwargs_mass2light=None,
        skypy_config=skypy_config,
        sky_area=sky_area,
        cosmo=cosmo,
    )

    kwargs_lens_cut = {
        "min_image_separation": 0.2,
        "max_image_separation": 10,
        "mag_arc_limit": {"F158": 25, "F106": 25, "F062": 25},
    }

    lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_cut)

    detectable = 0
    for candidate in lens_population:
        snr, _ = survey_sim.get_snr(gglens=candidate,
                                    band='F129',
                                    subtract_lens=True,
                                    mask_mult=1.)

        if snr > 10:
            detectable += 1

    lens_pop_counts.append(len(lens_population))
    detectable_counts.append(detectable)

print(f'Pre-SNR filtering: {lens_pop_counts}')
print(f'Post-SNR filtering: {detectable_counts}')
