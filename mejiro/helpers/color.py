from copy import deepcopy


# TODO some kind of wrapper to build color image given a lens class

# TODO method for linear color transformation

def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
