import os
import warnings

from astroquery.mast import Observations
from tqdm import tqdm

from package.utils import utils


def download_datasets(data_uri_list, download_path):
    utils.create_directory_if_not_exists(download_path)

    print('Ready to download ' + str(len(data_uri_list)) + ' files.\n')

    for data_uri in data_uri_list:
        print(
            'Dataset ' + str(data_uri_list.index(data_uri) + 1) + ' of ' + str(len(data_uri_list)) + ': ' + str(data_uri))

        filename = data_uri.split('/')[-1]

        result = Observations.download_file(data_uri, 
                                            local_path=os.path.join(download_path, filename))
        print(result)


def retrieve_datasets(dataset_list):
    data_uri_list = []
    skipped = []
    multiple = []
    num_skipped = 0
    num_multiple = 0

    for dataset in tqdm(dataset_list):

        data_products = retrieve_all_data_products(dataset.data_set_name)

        filtered_products = Observations.filter_products(data_products, 
                                                         productType=['SCIENCE'], 
                                                         calib_level=[3],
                                                         extension='fits', 
                                                         mrp_only=True,
                                                         productSubGroupDescription='DRZ')

        # check for no matching datasets and more than 1 matching dataset
        if len(filtered_products) == 0:
            message = 'No matching dataset found for ' + dataset.data_set_name
            warnings.warn(message)
            num_skipped += 1
            continue
        elif len(filtered_products) > 1:
            message = 'More than 1 matching dataset found for ' + dataset.data_set_name
            warnings.warn(message)
            num_multiple += 1
            continue
        else:
            data_uri_to_add = filtered_products[0]['dataURI']
            data_uri_list.append(data_uri_to_add)
            dataset.set_data_uri(data_uri_to_add)
            dataset.set_filename(data_uri_to_add.split('/')[-1])

    print('\n')
    print('Skipped: ' + str(num_skipped))
    print(skipped)
    print('\n')
    print('Multiple: ' + str(num_multiple))
    print(multiple)

    return data_uri_list, dataset_list


def retrieve_all_data_products(observation_id):
    single_obs = Observations.query_criteria(obs_id=observation_id)

    data_products = Observations.get_product_list(single_obs)

    return data_products
