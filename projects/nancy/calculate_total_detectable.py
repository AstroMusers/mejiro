import healpy as hp
import numpy as np
import os

detectable_counts = np.load('nancy_detectable_counts.npy')

nside = 5
npix = hp.nside2npix(nside)
pixel_area = hp.nside2pixarea(nside, degrees=True)
print(f"Area per pixel: {pixel_area:.2f} sq deg")

total = 0

for pix in range(npix):
    # for each pixel, that the detectable count per sq. deg. * area per pixel
    # to get the total detectable counts in that pixel
    total += (detectable_counts[pix] * pixel_area)

print(f"Total detectable counts over the sky: {total:.2f}")
