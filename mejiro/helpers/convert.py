

def mjy_to_counts(array, band):
    conversion_factor = get_mjy_to_counts_factor(band)
    return array * conversion_factor


def counts_to_mjy(array, band):
    conversion_factor = get_counts_to_mjy_factor(band)
    return array * conversion_factor


def get_mjy_to_counts_factor(band):
    mjy_to_counts = {
        'f106': 4765.629510460321,
        'f129': 4009.2786141694673,
        'f158': 3227.044734375456,
        'f184': 1969.1718210052677
    }

    return mjy_to_counts[band]


def get_counts_to_mjy_factor(band):
    counts_to_mjy = {
        'f106': 0.00020983586697309335,
        'f129': 0.0002494214286993753,
        'f158': 0.0003098810466888475,
        'f184': 0.000507827701642357
    }

    return counts_to_mjy[band]
