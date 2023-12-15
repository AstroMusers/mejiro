import pickle

from mejiro.lenses.lens import Lens


def unpickle_lens(pickle_path):
    with open(pickle_path, 'rb') as pickled_lens:
        unpickled = pickle.load(pickled_lens)

    kwargs_model = unpickled['kwargs_model']
    kwargs_params = unpickled['kwargs_params']

    return Lens(kwargs_model=kwargs_model, kwargs_params=kwargs_params)


def set_kwargs_params(kwargs_lens, kwargs_lens_light, kwargs_source):
    return {
        'kwargs_lens': kwargs_lens,
        'kwargs_lens_light': kwargs_lens_light,
        'kwargs_source': kwargs_source
    }


def set_kwargs_model(lens_model_list, lens_light_model_list, source_model_list):
    return {
        'lens_model_list': lens_model_list,
        'lens_light_model_list': lens_light_model_list,
        'source_light_model_list': source_model_list
    }
