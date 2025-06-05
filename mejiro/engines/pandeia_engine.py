import numpy as np
from lenstronomy.Util import data_util
from pandeia.engine.calc_utils import build_default_calc, build_default_source
from pandeia.engine.perform_calculation import perform_calculation
from tqdm import tqdm

from mejiro.engines.engine import Engine
from mejiro.instruments.roman import Roman


class PandeiaEngine(Engine):
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.lower() == 'roman':
            return {
                'num_samples': 10000,
                'calculation': {
                    'noise': {
                        'crs': True,
                        'dark': True,
                        'excess': False,  # Roman's detectors are H4RG which do not have excess noise parameters
                        'ffnoise': True,
                        'readnoise': True,
                        'scatter': False  # doesn't seem to have an effect
                    },
                    'effects': {
                        'saturation': True  # NB only has an effect for bright (>19mag) sources
                    }
                },
                'background': 'minzodi',
                'background_level': 'medium'  # 'benchmark', 'none
            }
        else:
            Engine.instrument_not_supported(instrument_name)


    @staticmethod
    def validate_engine_params(instrument_name, engine_params):
        if instrument_name.lower() == 'roman':
            if 'num_samples' not in engine_params.keys():
                engine_params['num_samples'] = PandeiaEngine.defaults('Roman')[
                    'num_samples']  # TODO is this necessary? doesn't GalSim do this?
                # TODO logging to inform user of default
            else:
                # TODO validate
                # TODO it doesn't like floats (e.g. 1e4), so make sure int
                pass
            if 'calculation' not in engine_params.keys():
                engine_params['calculation'] = PandeiaEngine.defaults('Roman')['calculation']
                # TODO logging to inform user of default
            else:
                if 'noise' not in engine_params['calculation'].keys():
                    engine_params['calculation']['noise'] = PandeiaEngine.defaults('Roman')['calculation']['noise']
                    # TODO logging to inform user of default
                else:
                    # TODO validate
                    pass
                if 'effects' not in engine_params['calculation'].keys():
                    engine_params['calculation']['effects'] = PandeiaEngine.defaults('Roman')['calculation']['effects']
                    # TODO logging to inform user of default
                else:
                    # TODO validate
                    pass
            if 'background' not in engine_params.keys():
                engine_params['background'] = PandeiaEngine.defaults('Roman')['background']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'background_level' not in engine_params.keys():
                engine_params['background_level'] = PandeiaEngine.defaults('Roman')['background_level']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            return engine_params
        else:
            Engine.instrument_not_supported(instrument_name)


    @staticmethod
    def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params=defaults('Roman'),
                        verbose=False, **kwargs):
        band = synthetic_image.band
        image = synthetic_image.image
        strong_lens = synthetic_image.strong_lens
        num_samples = engine_params['num_samples']
        oversample_factor = synthetic_image.oversample

        calc = build_default_calc('roman', 'wfi', 'imaging')

        # set scene size settings
        calc['configuration']['max_scene_size'] = synthetic_image.arcsec

        # set instrument
        calc['configuration']['instrument']['filter'] = synthetic_image.band.lower()

        # set detector
        # calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'  # TODO this causes error sometimes? but build_default_calc sets it to this value, so having/not having this line should make no difference... 
        calc['configuration']['detector'][
            'nresultants'] = 8  # resultant number 8 to achieve HLWAS total integration duration of 145.96 s; see https://roman-docs.stsci.edu/raug/astronomers-proposal-tool-apt/appendix/appendix-wfi-multiaccum-tables

        # set noise and detector effects
        calc['calculation'] = engine_params['calculation']

        # set Pandeia canned background
        calc['background'] = engine_params['background']
        calc['background_level'] = engine_params['background_level']

        # convert array from amp to counts/sec
        cps_array = PandeiaEngine._get_cps_array(strong_lens, image, num_samples, band, background=None)

        # convert array from counts/sec to astronomical magnitude
        mag_array = PandeiaEngine._convert_cps_to_magnitude(cps_array, band, synthetic_image.magnitude_zero_point)

        # add point sources to Pandeia input
        norm_wave = PandeiaEngine._get_norm_wave(band)
        if num_samples:
            calc, num_point_sources = PandeiaEngine._phonion_sample(calc, mag_array, synthetic_image, norm_wave, verbose)
        # TODO pass in an engine param to choose between sampling and grid methods
        elif oversample_factor:
            calc, num_point_sources = PandeiaEngine._phonion_grid(calc, mag_array, synthetic_image, norm_wave, verbose)
        else:
            raise Exception('Either provide num_samples to use sampling method or oversample_factor to use grid method')

        # if not suppress_output:
        #     print(f'Estimated calculation time: {estimate_calculation_time(num_point_sources)}')

        results = perform_calculation(calc)

        return results['2d']['detector'], None, None


    @staticmethod
    def _phonion_sample(calc, mag_array, synthetic_image, norm_wave, verbose):
        i = 0

        # loop over non-zero pixels, i.e. ignore pixels with no phonions
        for x, y in tqdm(np.argwhere(mag_array != np.inf), disable=not verbose):
            if i != 0:
                calc['scene'].append(build_default_source(geometry='point', telescope='roman'))

            # set brightness
            calc['scene'][i]['spectrum']['normalization']['norm_flux'] = mag_array[x][y]
            calc['scene'][i]['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
            calc['scene'][i]['spectrum']['normalization']['norm_wave'] = norm_wave
            calc['scene'][i]['spectrum']['normalization']['norm_waveunit'] = 'microns'
            calc['scene'][i]['spectrum']['normalization']['type'] = 'at_lambda'

            # calculate position. NB this returns the center of the pixel
            ra, dec = synthetic_image.pixel_grid.map_pix2coord(x=x, y=y)

            # set position
            calc['scene'][i]['position']['x_offset'] = dec
            calc['scene'][i]['position']['y_offset'] = -ra

            i += 1

        if verbose:
            print(f'Point source conversion complete: placed {i} point sources')

        # add an extra point source far out to force maximum scene size
        calc['scene'].append(build_default_source(geometry='point', telescope='roman'))
        calc['scene'][i]['position']['x_offset'] = 100
        calc['scene'][i]['position']['y_offset'] = 100

        return calc, i


    @staticmethod
    def _phonion_grid(calc, mag_array, synthetic_image, norm_wave, verbose):
        i = 0
        side, _ = mag_array.shape

        if verbose:
            print(f'Converting {mag_array.shape} array to point sources...')

        for row_number, row in tqdm(enumerate(mag_array), total=side, disable=not verbose):
            for item_number, item in enumerate(row):
                if i != 0:
                    calc['scene'].append(build_default_source(geometry='point', telescope='roman'))

                # set brightness
                calc['scene'][i]['spectrum']['normalization']['norm_flux'] = item
                calc['scene'][i]['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
                calc['scene'][i]['spectrum']['normalization']['norm_wave'] = norm_wave
                calc['scene'][i]['spectrum']['normalization']['norm_waveunit'] = 'microns'
                calc['scene'][i]['spectrum']['normalization']['type'] = 'at_lambda'

                # calculate position. NB this returns the center of the pixel
                ra, dec = synthetic_image.pixel_grid.map_pix2coord(x=item_number, y=row_number)

                # set position
                calc['scene'][i]['position']['x_offset'] = dec
                calc['scene'][i]['position']['y_offset'] = -ra
                # TODO clean up after confirming the above works
                # calc['scene'][i]['position']['x_offset'] = (item_number * (1 / 9) * (
                #         1 / oversample_factor)) + lens.ra_at_xy_0  # arcsec
                # calc['scene'][i]['position']['y_offset'] = (row_number * (1 / 9) * (
                #         1 / oversample_factor)) + lens.dec_at_xy_0  # arcsec

                i += 1

        if verbose:
            print(f'Point source conversion complete: placed {i} point sources')

        return calc, i


    @staticmethod
    def _get_cps_array(lens, array, num_samples, band, background):
        # normalize the image to convert it into a PDF
        sum = np.sum(array)
        normalized_array = array / sum

        # TODO why does this need to be here?
        # AttributeError: 'SampleStrongLens' object has no attribute 'lens_light_model_class'
        # so hasn't been defined yet...
        lens._set_classes()

        # # if including sky background, account for it; NB it must be in units of counts/sec/pixel
        # bkg_cps = 0
        # if background is not None:
        #     # calculate total flux due to background
        #     bkg_cps = np.sum(background)

        # get total flux so we know how bright to make each pixel (in counts/sec)
        counts_per_pixel = np.sum(array) / num_samples

        # turn array into probability distribution function and sample from it
        flattened = normalized_array.flatten()
        sample_indices = np.random.choice(a=flattened.size, p=flattened, size=num_samples)
        adjusted_indices = np.unravel_index(sample_indices, normalized_array.shape)
        adjusted_indices = np.array(list(zip(*adjusted_indices)))

        # build the sampled array
        reconstructed_array = np.zeros(array.shape)
        for x, y in adjusted_indices:
            reconstructed_array[x][y] += counts_per_pixel

        return reconstructed_array


    @staticmethod
    def _convert_cps_to_magnitude(array, band, zp):
        if zp is None:
            lenstronomy_roman_config = Roman(band=band.upper(), psf_type='PIXEL',
                                            survey_mode='wide_area').kwargs_single_band()  # band e.g. 'F106'
            magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')
        else:
            magnitude_zero_point = zp

        i = 0
        mag_array = np.zeros(array.shape)

        for row_number, row in enumerate(array):
            for item_number, item in enumerate(row):
                mag_array[row_number][item_number] = data_util.cps2magnitude(item, magnitude_zero_point)
                i += 1

        return mag_array


    @staticmethod
    def _get_norm_wave(band):
        band = band.upper()
        roman_params = PandeiaEngine._get_roman_params()
        filter_center_dict = roman_params.get_filter_centers()

        return filter_center_dict[band]
