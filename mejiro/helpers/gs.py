import datetime
import galsim
import random
import time
from astropy.coordinates import SkyCoord
from galsim import roman
from galsim import InterpolatedImage, Image

from mejiro.utils import util


def get_images(lens, arrays, bands, input_size, output_size, grid_oversample, detector=1, detector_pos=None, exposure_time=146, ra=30, dec=-30, seed=42):
    start = time.time()

    # was only one band provided as a string? or a list of bands?
    single_band = False
    if not isinstance(bands, list):
        single_band = True
        bands = [bands]

    # was only one array provided? or a list of arrays?
    single_array = False
    if not isinstance(arrays, list):
        single_array = True
        arrays = [arrays]

    # if a color image is desired, then three bands and three arrays should be provided
    if not single_band or not single_array:
        assert len(bands) == 3, 'For a color image, provide three bands'
        assert len(arrays) == 3, 'For a color image, provide three arrays'

     # make sure the arrays are square
    for array in arrays:
        assert array.shape[0] == array.shape[1], 'Input image must be square'

    # TODO they should also all have the same dimensions

    # create galsim rng
    rng = galsim.UniformDeviate(seed)

    # get wcs
    wcs_dict = get_wcs(ra, dec, date=None)

    # calculate sky backgrounds for each band
    bkgs = get_sky_bkgs(wcs_dict, bands, detector, exposure_time, num_pix=input_size)

    results = []
    for _, (band, array) in enumerate(zip(bands, arrays)):
        # get flux
        total_flux_cps = lens.get_total_flux_cps(band)  
        
        # get interpolated image
        interp = InterpolatedImage(Image(array, xmin=0, ymin=0), scale=0.11 / grid_oversample, flux=total_flux_cps * exposure_time)

        # generate PSF and convolve
        convolved = convolve(interp, band, detector, detector_pos, input_size, pupil_bin=1)

        # add sky background to convolved image
        final_image = convolved + bkgs[band]

        # integer number of photons are being detected, so quantize
        final_image.quantize()

        # add all detector effects
        galsim.roman.allDetectorEffects(final_image, prev_exposures=(), rng=rng, exptime=exposure_time)

        # make sure there are no negative values from Poisson noise generator
        final_image.replaceNegative()

        # get the array
        final_array = final_image.array

        # center crop to get rid of edge effects
        final_array = util.center_crop_image(final_array, (output_size, output_size))

        # divide through by exposure time to get in units of counts/sec/pixel
        final_array /= exposure_time

        results.append(final_array)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    
    return results, execution_time


def get_wcs(ra, dec, date=None):
    if date is None:
        date = datetime.datetime(year=2027, month=7, day=7, hour=0, minute=0, second=0)

    return _get_wcs_dict(ra, dec, date)


def get_random_hlwas_wcs(suppress_output=False):
    ra = random.uniform(15, 45)
    dec = random.uniform(-45, -15)

    # set observation datetime to midnight on July 7th, 2027 - this seems to be fine for all high galactic latitudes
    date = datetime.datetime(year=2027, month=7, day=7, hour=0, minute=0, second=0)

    if not suppress_output:
        print(f'RA: {ra}, DEC: {dec}')

    return _get_wcs_dict(ra, dec, date)


def _get_wcs_dict(ra, dec, date):
    skycoord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    ra_hms, dec_dms = skycoord.to_string('hmsdms').split(' ')

    ra_targ = galsim.Angle.from_hms(ra_hms)
    dec_targ = galsim.Angle.from_dms(dec_dms)
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)

    # NB targ_pos indicates the position to observe at the center of the focal plane array
    return roman.getWCS(world_pos=targ_pos, date=date)


def get_bandpass_key(band):
    band = band.upper()
    translate = {
        'F087': 'Z087',
        'F106': 'Y106',
        'F129': 'J129',
        'F158': 'H158',
        'F184': 'F184',
        'F149': 'W149'
    }
    return translate[band]


def get_bandpass(band):
    bandpass_key = get_bandpass_key(band)
    return roman.getBandpasses()[bandpass_key]


def get_random_detector(suppress_output=False):
    detector = random.randint(1, 18)
    if not suppress_output:
        print(f'Detector: {detector}')
    return detector


def get_random_detector_pos(input_size, suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    # we're adjusting inwards by the input_size because we want to make sure that the entire image fits on the detector, even before final cropping to remove any edge effects
    min = 4 + input_size
    max = 4092 - input_size

    x, y = random.randrange(min, max), random.randrange(min, max)
    
    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return galsim.PositionD(x, y)


def convolve(interp, band, detector, detector_position, input_size, pupil_bin=1):
    galsim_psf = roman.getPSF(detector, 
                              SCA_pos=detector_position, 
                              bandpass=None, 
                              wavelength=get_bandpass(band), 
                              pupil_bin=pupil_bin)

    # https://galsim-developers.github.io/GalSim/_build/html/composite.html#galsim.Convolve
    convolved = galsim.Convolve(interp, galsim_psf)

    # draw interpolated image at the final pixel scale
    im = galsim.ImageF(input_size, input_size, scale=0.11)  # NB setting dimensions to "input_size" because we'll crop down to "output_size" at the very end
    im.setOrigin(0, 0)

    return convolved.drawImage(im)


def get_sky_bkgs(wcs_dict, bands, detector, exposure_time, num_pix):
    bkgs = {}
    for band in bands:
        # get bandpass object
        bandpass = get_bandpass(band)

        # get wcs
        wcs = wcs_dict[detector]

        # build Image
        sky_image = galsim.ImageF(num_pix, num_pix, wcs=wcs)

        SCA_cent_pos = wcs.toWorld(sky_image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos, exptime=exposure_time)
        sky_level *= (1.0 + roman.stray_light_fraction)
        wcs.makeSkyImage(sky_image, sky_level)

        thermal_bkg = roman.thermal_backgrounds[get_bandpass_key(band)] * exposure_time
        sky_image += thermal_bkg

        bkgs[band] = sky_image

    return bkgs
