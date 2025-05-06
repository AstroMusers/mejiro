import numpy as np


def annular_mask(dimx, dimy, center, r_in, r_out):
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return (r_in <= distance_from_center) & \
        (distance_from_center <= r_out)
