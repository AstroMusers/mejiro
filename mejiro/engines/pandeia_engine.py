from pandeia.engine.calc_utils import build_default_calc, build_default_source
from pandeia.engine.perform_calculation import perform_calculation


def default_roman_engine_params():
    return {
        'max_scene_size': 5.0,  # arcsec
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
        }
    }


def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params=default_roman_engine_params(), verbose=False, **kwargs):
    calc = build_default_calc('roman', 'wfi', 'imaging')

    # set scene size settings
    calc['configuration']['max_scene_size'] = engine_params['max_scene_size']

    # set instrument
    calc['configuration']['instrument']['filter'] = synthetic_image.band.lower()

    # set detector
    calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'
    calc['configuration']['detector'][
        'nresultants'] = 8  # resultant number 8 to achieve HLWAS total integration duration of 145.96 s; see https://roman-docs.stsci.edu/raug/astronomers-proposal-tool-apt/appendix/appendix-wfi-multiaccum-tables
    
    # set noise and detector effects
    calc['calculation'] = engine_params['calculation']

    # set Pandeia canned background
    if sky_bkg:
        # calc['background'] = bkg.get_jbt_bkg(suppress_output)
        calc['background'] = 'minzodi'
        calc['background_level'] = 'high'  # 'benchmark'
    else:
        calc['background'] = 'none'

    


def validate_roman_engine_params(engine_params):
    pass

    # make sure num_samples is int
    # if not isinstance(num_samples, int):
    #     num_samples = int(num_samples)