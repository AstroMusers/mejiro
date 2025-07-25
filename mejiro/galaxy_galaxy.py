from astropy.cosmology import default_cosmology

from mejiro.strong_lens import StrongLens


class GalaxyGalaxy(StrongLens):

    def __init__(
            self,
            name,
            coords,
            kwargs_model,
            kwargs_params,
            physical_params={},
            use_jax=None
    ):
        super().__init__(name=name,
                         coords=coords, 
                         kwargs_model=kwargs_model,
                         kwargs_params=kwargs_params,
                         physical_params=physical_params,
                         use_jax=use_jax)
        
        # get redshifts
        self.z_source = kwargs_model['z_source']
        self.z_lens = kwargs_model['lens_redshift_list'][0]

    def get_einstein_radius(self):
        """
        Returns the Einstein radius from ``kwargs_lens`` if it has been suitably configured.

        Returns
        -------
        float
            The Einstein radius (``theta_E``) value extracted from ``self.kwargs_lens[0]``.

        Raises
        ------
        ValueError
            If the order of lens models is not as expected, or the main mass model does not have ``theta_E`` as a parameter, this method will not work and should be extended.
        """
        einstein_radius = self.kwargs_lens[0].get('theta_E', None)
        if einstein_radius is None:
            raise ValueError("Could not find `theta_E` in kwargs_lens")
        return einstein_radius

    def get_magnification(self):
        if 'magnification' not in self.physical_params:
            raise ValueError("Magnification not found in physical_params. Please provide 'magnification' in physical_params.")
        return self.physical_params['magnification']
        
    def get_image_positions(self, ignore_substructure=True):
        
        """
        Compute and return the extended source image positions.

        Parameters
        ----------
        ignore_substructure : bool, optional
            If True (default), ignores substructure in the lens model when computing image positions.
            If False, includes substructure in the calculation.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing arrays of x and y coordinates of the image positions in lenstronomy angle units (typically arcseconds).

        Notes
        -----
        - The source position is taken from the first element of ``kwargs_source``.
        - Uses lenstronomy's LensEquationSolver to solve the lens equation.
        - The solver is chosen automatically: "analytical" if supported by the lens model, otherwise "lenstronomy".
        - The search window and minimum distance are set based on the Einstein radius.
        - If a lens model realization exists and `ignore_substructure` is True, only the macromodel is used for the calculation.
        """
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver, analytical_lens_model_support

        source_x = self.kwargs_source[0]['center_x']
        source_y = self.kwargs_source[0]['center_y']

        if self.realization and ignore_substructure:
            lens_eqn_solver = LensEquationSolver(self.lens_model_macromodel)
            solver = "analytical" if analytical_lens_model_support(self.lens_model_list_macromodel) else "lenstronomy"
        else:
            lens_eqn_solver = LensEquationSolver(self.lens_model)
            solver = "analytical" if analytical_lens_model_support(self.lens_model_list) else "lenstronomy"
        
        return lens_eqn_solver.image_position_from_source(
            source_x,
            source_y,
            self.kwargs_lens,
            solver=solver,
            search_window=self.get_einstein_radius() * 6,
            min_distance=self.get_einstein_radius() * 6 / 200,
        )

    @staticmethod   
    def from_slsim(slsim_gglens, name=None, coords=None):
        # check that the input is reasonable
        if slsim_gglens.source_number != 1:
            raise ValueError("Only one source is supported for galaxy-galaxy lenses.")

        cosmo = slsim_gglens.cosmo
        z_lens = slsim_gglens.deflector_redshift
        z_source = slsim_gglens.source_redshift_list[0]

        # get the bands
        bands = [k.split("_")[1] for k in slsim_gglens.deflector._deflector._deflector_dict.keys() if k.startswith("mag_")]

        # get kwargs_model and kwargs_params
        kwargs_model, kwargs_params = slsim_gglens.lenstronomy_kwargs(band=bands[0])

        # add additional necessary key/value pairs to kwargs_model
        kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_params['kwargs_lens'])
        kwargs_model['source_redshift_list'] = [z_source]
        kwargs_model['cosmo'] = cosmo
        kwargs_model['z_source'] = z_source

        # populate magnitudes dictionary
        lens_mags, source_mags, lensed_source_mags = {}, {}, {}
        for band in bands:
            lens_mags[band] = slsim_gglens.deflector_magnitude(band)
            source_mags[band] = slsim_gglens.extended_source_magnitude(band, lensed=False)[0]
            lensed_source_mags[band] = slsim_gglens.extended_source_magnitude(band, lensed=True)[0]
        magnitudes = {
            'lens': lens_mags,
            'source': source_mags,
            'lensed_source': lensed_source_mags,
        }

        # populate physical parameters dictionary
        physical_params = {
            'lens_stellar_mass': slsim_gglens.deflector_stellar_mass(),
            'lens_velocity_dispersion': slsim_gglens.deflector_velocity_dispersion(),
            'magnification': slsim_gglens.extended_source_magnification()[0],
            'magnitudes': magnitudes
        }

        return GalaxyGalaxy(name=name,
                            coords=coords,
                            kwargs_model=kwargs_model,
                            kwargs_params=kwargs_params,
                            physical_params=physical_params)

    
class SampleGG(GalaxyGalaxy):
    """
    This is a simulated strong lens from `SLSim <https://github.com/LSST-strong-lensing/slsim>`__.
    """
    def __init__(self):
        name = 'SampleGG'
        coords = None
        kwargs_model = {
            'cosmo': default_cosmology.get(),
            'lens_light_model_list': ['SERSIC_ELLIPSE'],
            'lens_model_list': ['SIE', 'SHEAR', 'CONVERGENCE'],
            'lens_redshift_list': [0.2902115249535011, 0.2902115249535011, 0.2902115249535011],
            'source_light_model_list': ['SERSIC_ELLIPSE'],
            'source_redshift_list': [0.5876899931818929],
            'z_source': 0.5876899931818929,
        }
        kwargs_params = {
            'kwargs_lens': [
            {
                'center_x': -0.007876281728887604,
                'center_y': 0.010633393703246008,
                'e1': 0.004858808997848661,
                'e2': 0.0075210751726143355,
                'theta_E': 1.168082477232392
            },
            {
                'dec_0': 0,
                'gamma1': -0.03648819840013156,
                'gamma2': -0.06511863424492038,
                'ra_0': 0
            },
            {
                'dec_0': 0,
                'kappa': 0.06020941823541971,
                'ra_0': 0
            }
            ],
            'kwargs_lens_light': [
            {
                'R_sersic': 0.5300707454127908,
                'center_x': -0.007876281728887604,
                'center_y': 0.010633393703246008,
                'e1': 0.023377277902774978,
                'e2': 0.05349948216860632,
                'magnitude': None,  # if this doesn't get updated by the imaging process, want it to error out
                'n_sersic': 4.0
            }
            ],
            'kwargs_ps': None,
            'kwargs_source': [
            {
                'R_sersic': 0.1651633078964498,
                'center_x': 0.30298310338567075,
                'center_y': -0.3505004565139597,
                'e1': -0.06350855238708408,
                'e2': -0.08420760408362458,
                'magnitude': None,  # if this doesn't get updated by the imaging process, want it to error out
                'n_sersic': 1.0
            }
            ]
        }
        magnitudes = {
            'lens': {
                'F062': 17.9,
                'F087': 17.7,
                'F106': 17.5,
                'F129': 17.3,
                'F158': 17.1,
                'F184': 17.0,
                'F146': 17.1,
                'F213': 16.9,
                'B': 17,
                'FUV': 17,
                'H': 17,
                'I': 17,
                'J': 17,
                'K': 17,
                'NUV': 17,
                'R': 17,
                'U': 17,
                'V': 17
            },
            'source': {
                'F062': 21.9,
                'F087': 21.7,
                'F106': 21.4,
                'F129': 21.1,
                'F158': 20.9,
                'F184': 20.5,
                'F146': 21.0,
                'F213': 20.4,
                'B': 20,
                'FUV': 20,
                'H': 20,
                'I': 20,
                'J': 20,
                'K': 20,
                'NUV': 20,
                'R': 20,
                'U': 20,
                'V': 20
            }
        }
        physical_params = {
            'magnitudes': magnitudes,
            'main_halo_mass': 1e13
        }
        
        super().__init__(name=name,
                         coords=coords,
                         kwargs_model=kwargs_model, 
                         kwargs_params=kwargs_params,
                         physical_params=physical_params,
                         use_jax=[True, True, True])


class SampleBELLS(GalaxyGalaxy):
    """
    This system, SDSSJ1159-0007, from the BELLS sample (`Brownstein et al. 2012 <https://doi.org/10.1088/0004-637X/744/1/41>`__, `Shu et al. 2016 <https://doi.org/10.3847/1538-4357/833/2/264>`__) was modeled by `Tan et al. (2024) <https://doi.org/10.1093/mnras/stae884>`__.
    """
    def __init__(self):
        name = 'SDSSJ1159-0007'
        coords = None
        kwargs_model = {
            'cosmo': default_cosmology.get(),
            'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'],
            'lens_model_list': ['EPL', 'SHEAR_GAMMA_PSI'],
            'lens_redshift_list': [0.58, 0.58],
            'source_light_model_list': ['SHAPELETS', 'SERSIC_ELLIPSE'],
            'source_redshift_list': [1.35, 1.35],
            'z_source': 1.35,
        }
        kwargs_params = {
            'kwargs_lens': [
            {
                'center_x': 0.0008993530339094509,
                'center_y': 0.004386561103657995,
                'e1': -0.13834112391265768,
                'e2': -0.04767716373059794,
                'gamma': 2.6509527320161292,
                'theta_E': 0.6745078855269103
            },
            {
                'dec_0': 0,
                'gamma_ext': 0.07157053126083526,
                'psi_ext': 1.2573502199996038,
                'ra_0': 0
            }
            ],
            'kwargs_lens_light': [
            {
                'R_sersic': 4.9386180836201605,
                'amp': 0.8605269799538531,
                'center_x': 0.0008993530339094509,
                'center_y': 0.004386561103657995,
                'e1': -0.02858565563884006,
                'e2': -0.017832976824861253,
                'n_sersic': 4.0
            },
            {
                'R_sersic': 0.25019004614994445,
                'amp': 119.59290072500922,
                'center_x': 0.0008993530339094509,
                'center_y': 0.004386561103657995,
                'e1': -0.02858565563884006,
                'e2': -0.017832976824861253,
                'n_sersic': 1.0
            }
            ],
            'kwargs_source': [
            {
                'amp': [
                1.97341789e+02, -7.25473768e+00, 1.43093020e+01, 5.67273098e+01,
                -3.49991809e+01, 2.37584187e+01, 6.37978766e+01, -3.04488121e+01,
                -1.98933165e+01, -1.11134993e+01, -3.08509426e+00, 1.39713699e+01,
                -1.63025361e+01, 3.79437438e+01, -2.44066712e+01, -5.16339322e+01,
                4.97848064e-02, -2.03668641e+01, 2.27456554e+01, -1.86275814e+01,
                1.80053583e+01, -3.95527313e+00, 6.28643274e+00, 1.19397493e+01,
                9.68964080e+00, 1.60624553e+01, 3.80978139e-01, 3.75319976e+01
                ],
                'beta': 0.09772413472924785,
                'center_x': -0.2355420617897601,
                'center_y': 0.16724538194630992,
                'n_max': 6
            },
            {
                'R_sersic': 0.4962690147124496,
                'amp': 7.4711400766341,
                'center_x': -0.2355420617897601,
                'center_y': 0.16724538194630992,
                'e1': -0.01917665283933903,
                'e2': -0.46613640509010695,
                'n_sersic': 1.0218574567182759
            }
            ]
        }
        physical_params = {
            'main_halo_mass': 10 ** 13.3
        }
        
        super().__init__(name=name,
                         coords=coords,
                         kwargs_model=kwargs_model, 
                         kwargs_params=kwargs_params,
                         physical_params=physical_params,
                         use_jax=[True, False])


class SampleSL2S(GalaxyGalaxy):
    """
    This system, SL2SJ1411+5651, from the SL2S sample (`Gavazzi et al. 2012 <https://doi.org/10.1088/0004-637X/761/2/170>`__) was modeled by `Tan et al. (2024) <https://doi.org/10.1093/mnras/stae884>`__.
    """
    def __init__(self):
        name = 'SL2SJ1411+5651'
        coords = None
        kwargs_model = {
            'cosmo': default_cosmology.get(),
            'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE', 'SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'],
            'lens_model_list': ['EPL', 'SHEAR_GAMMA_PSI'],
            'lens_redshift_list': [0.32, 0.32],
            'source_light_model_list': ['SHAPELETS', 'SERSIC_ELLIPSE', 'SHAPELETS', 'SERSIC_ELLIPSE'],
            'source_redshift_list': [1.42, 1.42, 1.42, 1.42],
            'z_source': 1.42,
        }
        kwargs_params = {
            'kwargs_lens': [
            {
                'center_x': 0.005181151321089338,
                'center_y': 0.013856469382939428,
                'e1': -0.05207199553562576,
                'e2': 0.04251551575381039,
                'gamma': 1.7285382185059783,
                'theta_E': 0.9190348632911911
            },
            {
                'dec_0': 0,
                'gamma_ext': 0.056662622162326884,
                'psi_ext': 1.4966747510338838,
                'ra_0': 0
            }
            ],
            'kwargs_lens_light': [
            {
                'R_sersic': 4.978232709615057,
                'amp': 0.4054511532032136,
                'center_x': 0.005181151321089338,
                'center_y': 0.013856469382939428,
                'e1': -0.11563613140066009,
                'e2': 0.020870342707182965,
                'n_sersic': 4.0
            },
            {
                'R_sersic': 0.2920889625760988,
                'amp': 52.484415076554924,
                'center_x': 0.005181151321089338,
                'center_y': 0.013856469382939428,
                'e1': -0.11563613140066009,
                'e2': 0.020870342707182965,
                'n_sersic': 1.0
            },
            {
                'R_sersic': 4.978232709615057,
                'amp': 1.1734686974816284,
                'center_x': 0.005181151321089338,
                'center_y': 0.013856469382939428,
                'e1': -0.11563613140066009,
                'e2': 0.020870342707182965,
                'n_sersic': 4.0
            },
            {
                'R_sersic': 0.2920889625760988,
                'amp': 205.2335215947435,
                'center_x': 0.005181151321089338,
                'center_y': 0.013856469382939428,
                'e1': -0.11563613140066009,
                'e2': 0.020870342707182965,
                'n_sersic': 1.0
            }
            ],
            'kwargs_source': [
            {
                'amp': [
                208.23804336,  75.24750135,  36.1395493 , -68.34135676,  80.83477441,
                44.59064691,  85.83113614,  51.44761156,  52.6408002 ,  20.00682991,
                63.22746006, -28.27498342,  63.55928184, -18.5433216 ,  -9.16220457,
                -5.80255161, -14.05993575,  -0.68008446,  24.11529423,   3.17577014,
                32.62302734, -16.34963974,  35.38465367,   6.63822638,  60.05857453,
                10.21890193,   6.32843717,  27.39353694,  63.91061969,  39.51415767,
                27.76777897,  14.56705787,  47.21652381,  12.22472449,  -4.73374628,
                -22.17029146,  12.07932979, -12.40503004, -28.33298552, -80.12889905,
                4.75413459, -48.07947372, -24.21941197,   2.3321554 ,  17.62417157,
                -11.13478151,  13.77947375,  49.25002374,  -8.04890207,   0.66989539,
                -5.62759948,   2.8662009 ,  24.90431891,  53.14985793,  39.29554171,
                -20.1348302 ,   9.58448674,  45.76348456,  61.66455832,   9.80683069,
                37.25634355,  -3.69236946,  26.61791072,  22.11509477,   1.25807807,
                -17.86006758
                ],
                'beta': 0.02003743857202428,
                'center_x': -0.008727853862921346,
                'center_y': -0.03257850360121771,
                'n_max': 10
            },
            {
                'R_sersic': 0.4980735751847538,
                'amp': 0.38563820199937265,
                'center_x': -0.008727853862921346,
                'center_y': -0.03257850360121771,
                'e1': 0.22729141994823104,
                'e2': 0.21694541553705607,
                'n_sersic': 7.98955494278538
            },
            {
                'amp': [
                -2.51747041e+02,  1.65220759e+01,  1.64853728e+01, -3.38342277e+01,
                -3.77933389e+01,  3.39527820e+01, -1.29859861e+01, -2.90351117e+00,
                8.95879362e-01, -7.79902274e+00, -4.35131501e+01, -1.34014392e+01,
                -2.83720759e+01,  4.61556777e+00, -4.14787215e+01,  8.05963711e+00,
                -6.90196599e-01,  6.01721502e+00,  6.62749644e+00,  5.83648347e+00,
                1.24981425e+01, -1.28990804e+01, -7.49783340e+00, -2.41378682e+00,
                -8.34251618e+00,  1.02815970e+01, -1.35194966e+01,  1.72782202e+01,
                -5.01631358e+00, -7.97659881e+00, -2.21678077e+00,  1.22151852e+00,
                3.62359590e+00, -2.95358687e+00,  1.57051856e+00, -1.15079422e+01,
                -1.82955381e+01, -3.14337804e+00, -2.11702411e+01, -2.20565799e+00,
                -1.43500962e+01,  5.16904826e+00, -1.40909735e+01,  1.38665374e+00,
                -2.28747596e+01, -1.45979446e+00,  9.94737977e-01,  2.62159653e+00,
                5.23724824e+00,  1.83356330e+00,  7.45563692e+00,  1.99943672e-01,
                9.43579490e+00, -1.73761695e+00,  2.24692711e+01, -2.14936748e+01,
                -1.08441849e+01, -6.10671160e+00, -2.65210128e+00, -5.01023462e+00,
                -7.43264336e+00,  5.60630789e+00, -4.56715656e+00,  8.87027537e+00,
                -4.92079225e+00,  9.61319007
                ],
                'beta': 0.06933938109343586,
                'center_x': -0.008727853862921346,
                'center_y': -0.03257850360121771,
                'n_max': 10
            },
            {
                'R_sersic': 0.4980735751847538,
                'amp': 4.70268704612566,
                'center_x': -0.008727853862921346,
                'center_y': -0.03257850360121771,
                'e1': 0.22729141994823104,
                'e2': 0.21694541553705607,
                'n_sersic': 3.9577354181414575
            }
            ]
        }
        physical_params = {
            'main_halo_mass': 10 ** 13.3
        }
        
        super().__init__(name=name,
                         coords=coords,
                         kwargs_model=kwargs_model, 
                         kwargs_params=kwargs_params,
                         physical_params=physical_params,
                         use_jax=[True, False])
        