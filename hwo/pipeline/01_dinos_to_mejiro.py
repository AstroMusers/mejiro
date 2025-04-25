import os
import sys
import yaml
from glob import glob
from pprint import pprint

import numpy as np
import astropy.cosmology as astropy_cosmo
import pandas as pd
from dolphin.analysis.output import Output
from tqdm import tqdm


def main():
    # enable use of local modules
    repo_dir = '/grad/bwedig/mejiro'
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.galaxy_galaxy import GalaxyGalaxy
    from mejiro.utils import util

    # set up directories to save output to
    data_dir = '/data/bwedig/mejiro'
    save_dir = os.path.join(data_dir, 'hwo', 'dinos')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # lists of dinos systems to process
    Double_SLACS = ['SDSSJ0029-0055', 'SDSSJ0037-0942', 'SDSSJ0819+4534', 'SDSSJ0903+4116', 'SDSSJ0936+0913',
                'SDSSJ0959+0410', 'SDSSJ1134+6027', 'SDSSJ1204+0358', 'SDSSJ1213+6708', 'SDSSJ1218+0830',
                'SDSSJ1531-0105', 'SDSSJ1621+3931', 'SDSSJ1627-0053', 'SDSSJ2302-0840']

    Single_SLACS = ['SDSSJ0008-0004', 'SDSSJ0252+0039', 'SDSSJ0330-0020', 'SDSSJ0728+3835', 'SDSSJ0737+3216',
                    'SDSSJ0912+0029', 'SDSSJ1023+4230', 'SDSSJ1100+5329', 'SDSSJ1112+0826', 'SDSSJ1250+0523',
                    'SDSSJ1306+0600', 'SDSSJ1313+4615', 'SDSSJ1402+6321', 'SDSSJ1630+4520', 'SDSSJ1636+4707',
                    'SDSSJ2238-0754', 'SDSSJ2300+0022', 'SDSSJ2303+1422', 'SDSSJ2343-0030', 'SDSSJ2347-0005']

    Double_SL2S = ['SL2SJ0208-0714', 'SL2SJ0219-0829', 'SL2SJ1427+5516']
    Single_SL2S = ['SL2SJ0214-0405', 'SL2SJ0217-0513', 'SL2SJ0225-0454', 'SL2SJ0226-0406', 'SL2SJ0226-0420',
                'SL2SJ0232-0408', 'SL2SJ0849-0251', 'SL2SJ0849-0412', 'SL2SJ0858-0143', 'SL2SJ0901-0259',
                'SL2SJ0904-0059', 'SL2SJ0959+0206', 'SL2SJ1358+5459', 'SL2SJ1359+5535', 'SL2SJ1401+5544',
                'SL2SJ1402+5505', 'SL2SJ1405+5243', 'SL2SJ1406+5226', 'SL2SJ1411+5651', 'SL2SJ1420+5630',
                'SL2SJ2214-1807']

    Double_BELLS = ['SDSSJ0801+4727', 'SDSSJ0944-0147', 'SDSSJ1234-0241', 'SDSSJ1349+3612', 'SDSSJ1542+1629',
                    'SDSSJ1631+1854', 'SDSSJ2125+0411']
    Single_BELLS = ['SDSSJ0151+0049', 'SDSSJ0747+5055', 'SDSSJ0830+5116', 'SDSSJ1159-0007', 'SDSSJ1215+0047',
                    'SDSSJ1221+3806', 'SDSSJ1318-0104', 'SDSSJ1337+3620', 'SDSSJ1352+3216', 'SDSSJ1545+2748',
                    'SDSSJ1601+2138', 'SDSSJ2303+0037']

    slacs = Double_SLACS + Single_SLACS
    sl2s = Double_SL2S + Single_SL2S
    bells = Double_BELLS + Single_BELLS

    dinos_df = pd.read_csv(os.path.join(repo_dir, 'hwo', 'data', 'dinos_i_tan_et_al_2024', 'dinos_i.csv'))

    for system_name in tqdm(slacs + sl2s + bells):
        try:
            # get system information from CSV
            row = dinos_df[dinos_df['Lens system'] == system_name].iloc[0]

            output = Output('/nfsdata1/bwedig/dinos_i_outputs')
            _ = output.load_output(system_name, model_id='dinos_i')
            _ = output.plot_model_overview(lens_name=system_name, model_id='dinos_i')

            # create StrongLens
            if row['z_lens'] == '--' or row['z_source'] == '--':
                continue
            z_lens = float(row['z_lens'])
            z_source = float(row['z_source'])

            kwargs_lens = output.kwargs_result['kwargs_lens']
            kwargs_lens_light = output.kwargs_result['kwargs_lens_light']
            kwargs_source = output.kwargs_result['kwargs_source']

            kwargs_params = {
                'kwargs_lens': kwargs_lens,
                'kwargs_lens_light': kwargs_lens_light,
                'kwargs_source': kwargs_source
            }

            kwargs_model = {
                'cosmo': astropy_cosmo.default_cosmology.get(),
                'lens_light_model_list': output.model_settings['model']['lens_light'],
                'lens_model_list': output.model_settings['model']['lens'],
                'lens_redshift_list': [z_lens] * len(kwargs_lens),
                'source_light_model_list': output.model_settings['model']['source_light'],
                'source_redshift_list': [z_source] * len(kwargs_source),
                'z_source': z_source,
                'z_source_convention': 5.
            }
            assert len(kwargs_model['lens_model_list']) == len(kwargs_model['lens_redshift_list'])
            assert len(kwargs_model['source_light_model_list']) == len(kwargs_model['source_redshift_list'])

            strong_lens = GalaxyGalaxy(name=system_name,
                           coords=None,  # TODO TEMP
                           kwargs_model=kwargs_model,
                           kwargs_params=kwargs_params)
            
            util.pickle(os.path.join(save_dir, f'{system_name}.pkl'), strong_lens)
        except Exception as e:
            print(f'Error processing {system_name}: {e}')
            continue


if __name__ == '__main__':
    main()
