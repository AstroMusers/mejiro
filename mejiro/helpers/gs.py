import datetime
import galsim
import random
from astropy.coordinates import SkyCoord
from galsim import roman
from galsim import InterpolatedImage, Image


def get_random_hlwas_wcs(suppress_output=False):
    ra = random.uniform(15, 45)
    dec = random.uniform(-45, -15)

    # set observation datetime to midnight on July 7th, 2027 - this seems to be fine for all high galactic latitudes
    date = datetime.datetime(year=2027, month=7, day=7, hour=0, minute=0, second=0)

    skycoord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    ra_hms, dec_dms = skycoord.to_string('hmsdms').split(' ')

    ra_targ = galsim.Angle.from_hms(ra_hms)
    dec_targ = galsim.Angle.from_dms(dec_dms)
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)

    # NB targ_pos indicates the position to observe at the center of the focal plane array
    wcs_dict = roman.getWCS(world_pos=targ_pos, date=date)

    if not suppress_output:
        print(f'RA: {ra}, DEC: {dec}')

    return wcs_dict


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


def get_random_detector_pos(suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    x, y = random.randrange(4, 4092), random.randrange(4, 4092)
    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return galsim.PositionD(x, y)


def convolve(interp, band, detector, detector_position, num_pix, pupil_bin=1):
    galsim_psf = roman.getPSF(detector, 
                              SCA_pos=detector_position, 
                              bandpass=None, 
                              wavelength=get_bandpass(band), 
                              pupil_bin=pupil_bin)

    # https://galsim-developers.github.io/GalSim/_build/html/composite.html#galsim.Convolve
    convolved = galsim.Convolve(interp, galsim_psf)

    # draw interpolated image
    im = galsim.ImageF(num_pix, num_pix, scale=0.11)
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
