def divide_up_sca(sides):
    """
    Divides the SCA into a grid of sub-arrays and calculates the center coordinates of each sub-array.

    Parameters
    ----------
    sides : int
        The number of divisions along one side of the SCA. Must be a positive integer.

    Returns
    -------
    np.ndarray
        An array of shape (sides * sides, 2) containing the (x, y) coordinates of the centers of the sub-arrays.

    Raises
    ------
    AssertionError
        If `sides` is not a positive integer.
    """
    sides = int(sides)
    assert sides > 0, "Input sides must be a positive integer."

    sub_array_size = 4088 / sides
    centers = []

    for i in range(sides):
        for j in range(sides):
            center_x = int(round((i + 0.5) * sub_array_size))
            center_y = int(round((j + 0.5) * sub_array_size))
            centers.append((center_x, center_y))

    return centers


def get_sca_string(sca):
    """
    Converts an input value to a standardized SCA string format.

    Parameters
    ----------
    sca : int, float, or str
        The input value to be converted. It can be:
        - int: An integer value.
        - float: A floating-point value.
        - str: A string that may start with 'SCA' or be a numeric string.

    Returns
    -------
    str
        A string in the format 'SCAxx', where 'xx' is a zero-padded integer.

    Raises
    ------
    ValueError
        If the input value is not of type int, float, or str, or if the string
        format is not recognized.
    """
    if type(sca) is int:
        return f'SCA{str(sca).zfill(2)}'
    elif type(sca) is float:
        return f'SCA{str(int(sca)).zfill(2)}'
    elif type(sca) is str:
        # might provide int 1 or string 01 or string SCA01
        if sca.startswith('SCA'):
            end = sca[3:]
            # TODO check that `end` is valid, e.g., not SCA01X
            return f'SCA{str(int(end)).zfill(2)}'
        else:
            return f'SCA{str(int(sca)).zfill(2)}'
    else:
        raise ValueError(f"SCA {sca} not recognized. Must be int, float, or str.")


def get_sca_int(sca):
    """
    Convert a given SCA value to an integer.

    Parameters
    ----------
    sca : int, float, or str
        The SCA value to be converted. It can be an integer, a float, or a string.
        If it is a string, it can optionally start with 'SCA'.

    Returns
    -------
    int
        The integer representation of the SCA value.

    Raises
    ------
    ValueError
        If the input `sca` is not of type int, float, or str, or if the string
        format is not recognized.
    """
    if type(sca) is int:
        return sca
    if type(sca) is float:
        return int(sca)
    elif type(sca) is str:
        if sca.startswith('SCA'):
            return int(sca[3:])
        else:
            return int(sca)
    else:
        raise ValueError(f"SCA {sca} not recognized. Must be int, float, or str.")


# TODO consider making all methods which might call this one methods on the class that only call this method if a flag on the class (e.g., override) is False. this way, scripts can instantiate Roman() with override=True and then avoid running this method every single time
def translate_band(input):
    """
    Translates a given band name or alias to its corresponding Roman filter band.
    This function takes an input band name or alias and translates it to the 
    corresponding Roman filter band. The function supports various aliases 
    for each band, including ground-based filter names.

    Parameters
    ----------
    input : str
        The input band name or alias to be translated.

    Returns
    -------
    str
        The corresponding Roman filter band name.

    Raises
    ------
    ValueError
        If the input band name or alias is not recognized.
        
    Examples
    --------
    >>> translate_band('R')
    'F062'
    >>> translate_band('Z087')
    'F087'
    >>> translate_band('H/K')
    'F184'
    >>> translate_band('unknown')
    Traceback (most recent call last):
        ...
    ValueError: Band UNKNOWN not recognized. Valid bands (and aliases) are {'F062': ['F062', 'R', 'R062'], 'F087': ['F087', 'Z', 'Z087'], 'F106': ['F106', 'Y', 'Y106'], 'F129': ['F129', 'J', 'J129'], 'F158': ['F158', 'H', 'H158'], 'F184': ['F184', 'H/K'], 'F146': ['F146', 'WIDE', 'W146'], 'F213': ['F213', 'KS', 'K213']}.
    """
    # some folks are referring to Roman filters using the corresponding ground-based filters (r, z, Y, etc.)
    options_dict = {
        'F062': ['F062', 'R', 'R062'],
        'F087': ['F087', 'Z', 'Z087'],
        'F106': ['F106', 'Y', 'Y106'],
        'F129': ['F129', 'J', 'J129'],
        'F158': ['F158', 'H', 'H158'],
        'F184': ['F184', 'H/K'],
        'F146': ['F146', 'WIDE', 'W146'],
        'F213': ['F213', 'KS', 'K213']
    }

    # check if input is already a valid band
    if input in options_dict.keys():
        return input

    # capitalize and remove spaces
    input = input.upper().replace(' ', '')

    for band, possible_names in options_dict.items():
        if input in possible_names:
            return band

    # if haven't returned yet, alias wasn't found
    raise ValueError(f"Band {input} not recognized. Valid bands (and aliases) are {options_dict}.")
