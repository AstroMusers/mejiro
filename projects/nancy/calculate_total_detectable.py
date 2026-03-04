import healpy as hp
import numpy as np
import os


dev = True

if dev:
    count_file = 'nancy_detectable_counts_dev.npy'
else:    
    count_file = 'nancy_detectable_counts.npy'

# load detectable counts
detectable_counts = np.load(count_file)

if dev:
    nside = 5
else:
    nside = 25

npix = hp.nside2npix(nside)
pixel_area = hp.nside2pixarea(nside, degrees=True)
print(f"Area per pixel: {pixel_area:.2f} sq deg")

total = 0
area = 0

# for each pixel, that the detectable count per sq. deg. * area per pixel
# to get the total detectable counts in that pixel

for pix in range(npix):
    detectable_rate = detectable_counts[pix]  # counts per sq. deg.

    if not np.isnan(detectable_rate):
        total += (detectable_rate * pixel_area)
        area += pixel_area

print(f"Total detectable counts over the sky: {total:.2f}")
print(f"Detectable counts per sq. deg. on average: {total / area:.2f}")
